# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import contextlib
import random
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, LlamaTokenizer, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel
from .beats.BEATs import BEATsConfig, BEATs
from .utils import StoppingCriteriaSub


class SALMONN(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        # Use bfloat16 on GPU - same memory as fp16 but doesn't need GradScaler
        # (larger dynamic range). float32 on CPU for compatibility.
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=model_dtype,
            )
        logging.info(f'LLaMA dtype: {model_dtype}')

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        assert whisper_path
        logging.info('Loading Whisper Model')
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")
        
        if self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()
                logging.info("freeze BEATs")

        if self.use_speech_Qformer:
            if self.beats_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            if self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False,
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False,
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples, verbose=False, output_attentions=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_attentions=output_attentions,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        out = {"loss": loss}
        if output_attentions:
            out["attentions"] = torch.stack(outputs.attentions, dim=0)
            mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            out["attentions_meta"] = {
                "to_regress_tokens": to_regress_tokens.input_ids,
                "start_regress": atts_bos.shape[1] + speech_embeds.shape[1],
                "speech_len": speech_embeds.shape[1],
                "mask_4d": ~torch.tril(mask_4d.repeat(1, 1, mask_4d.shape[-1], 1)),
            }
        return out

    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model


class CoTSALMONN(SALMONN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(42)
        
    def forward(self, samples, verbose=False, focus_on_conclusion_probability=0.33):
        if random.random() > focus_on_conclusion_probability or not self.training:
            return super(CoTSALMONN, self).forward(samples, verbose)

        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        # Build the conclusion mask
        conclusion_end_token_id = self.llama_tokenizer("</CONCLUSION>", add_special_tokens=False).input_ids
        conclusion_mask = torch.zeros(*to_regress_tokens.input_ids.shape, dtype=torch.bool, device=to_regress_tokens.input_ids.device)
        for i, t in enumerate(text):
            conclusion_tokens = self.llama_tokenizer(t.split("<CONCLUSION>")[1], add_special_tokens=False).input_ids
            padding_length = (1 - to_regress_tokens.attention_mask[i]).sum()
            conclusion_mask[i, -(padding_length+len(conclusion_tokens)-1):-(padding_length+len(conclusion_end_token_id)+1)] = True

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill(~conclusion_mask, -100)
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}


class ReverseSALMONN(SALMONN):
    def __init__(self, alpha=1.0, reverse_prob=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.reverse_prob = reverse_prob

    @classmethod
    def from_config(cls, config):
        # Get ReverseSALMONN-specific params
        alpha = config.get("alpha", 1.0)
        reverse_prob = config.get("reverse_prob", 0.5)
        
        # Get all base SALMONN params
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            alpha=alpha,
            reverse_prob=reverse_prob,
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load ReverseSALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model

    def _prompt_wrap_with_att_masks(self, embeds, atts, prompt, multi_prompt=False):
        """ReverseSALMONN-specific: returns (embeds, atts, p_before_atts, p_after_atts).
        Do not override prompt_wrap - base class expects 2 return values when super().forward() is called.
        """
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False,
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False,
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts, p_before_tokens.attention_mask, p_after_tokens.attention_mask
        else:
            return embeds, atts, None, None
        
    def forward(self, samples, verbose=False):
        if not self.training:
            return super().forward(samples, verbose)

        if random.random() > self.reverse_prob:
            return super().forward(samples, verbose)

        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts, p_before_atts, p_after_atts = self._prompt_wrap_with_att_masks(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # Reverse the sequence
        reversed_speech_embeds = speech_embeds.flip(dims=[1])
        reversed_speech_atts = speech_atts.flip(dims=[1])
        reversed_p_before_atts = p_before_atts.flip(dims=[1])
        reversed_p_after_atts = p_after_atts.flip(dims=[1])

        # Extract the audio embeds
        audio_embeds = reversed_speech_embeds[:, reversed_p_after_atts.shape[1]:-reversed_p_before_atts.shape[1]]
        audio_atts = reversed_speech_atts[:, reversed_p_after_atts.shape[1]:-reversed_p_before_atts.shape[1]]

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        batch_size = to_regress_tokens.input_ids.shape[0]
        device = to_regress_tokens.input_ids.device

        # Reverse the sequence
        reversed_to_regress_tokens = to_regress_tokens.input_ids.flip(dims=[1])
        reversed_to_regress_tokens_atts = to_regress_tokens.attention_mask.flip(dims=[1])

        reversed_to_regress_embeds = self.llama_model.model.embed_tokens(reversed_to_regress_tokens) if not self.lora else self.llama_model.model.model.embed_tokens(reversed_to_regress_tokens)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=device,
        )

        reversed_inputs_embeds = torch.cat([reversed_to_regress_embeds, reversed_speech_embeds, bos_embeds], dim=1)
        reversed_attention_mask = torch.cat([reversed_to_regress_tokens_atts, reversed_speech_atts, atts_bos], dim=1)

        # Create base 4D causal mask
        total_len = to_regress_tokens.input_ids.shape[1] + speech_embeds.shape[1] + bos_embeds.shape[1]
        mask_4d = torch.ones([batch_size, 1, total_len, total_len], dtype=speech_embeds.dtype, device=device)
        mask_4d = torch.tril(mask_4d, diagonal=0)
        # Prompt + bos tokens are always attended to
        mask_4d[:, 0, :, -bos_embeds.shape[1]:] = 1.0
        mask_4d[:, 0, :, -(reversed_p_before_atts.shape[1]+bos_embeds.shape[1]):-bos_embeds.shape[1]] = reversed_p_before_atts.unsqueeze(1)
        mask_4d[:, 0, :, -(reversed_speech_atts.shape[1]+bos_embeds.shape[1]):-(reversed_speech_atts.shape[1]-reversed_p_after_atts.shape[1]+bos_embeds.shape[1])] = reversed_p_after_atts.unsqueeze(1)
        mask_4d[:, 0] *= reversed_attention_mask.unsqueeze(1)
        # Conclusion tokens attended to audio tokens
        conclusion_texts = []
        for t in text:
            match = re.search(rf"(<CONCLUSION>.*?</CONCLUSION>)", t, re.DOTALL)
            conclusion_text = match.group(1).strip()
            conclusion_texts.append(conclusion_text)
        conclusion_tokens = self.llama_tokenizer(
            conclusion_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to(device)
        reversed_conclusion_input_ids = conclusion_tokens.input_ids.flip(dims=[1])
        reversed_conclusion_atts = conclusion_tokens.attention_mask.flip(dims=[1])
        padding_len = ((1.0 - reversed_conclusion_atts).sum(dim=1) + (1.0 - reversed_to_regress_tokens_atts).sum(dim=1)).int()
        for i in range(batch_size):
            mask_4d[i, 0, padding_len[i]:padding_len[i]+reversed_conclusion_input_ids.shape[1], -(audio_atts.shape[1]+reversed_p_before_atts.shape[1]+bos_embeds.shape[1]):-(reversed_p_before_atts.shape[1]+bos_embeds.shape[1])] = reversed_conclusion_atts[i].unsqueeze(1)

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min))

        # Create position ids
        position_ids = torch.arange(total_len, device=device).unsqueeze(0).repeat(batch_size, 1).flip(dims=[1])

        # Create targets for the regress tokens
        targets = reversed_to_regress_tokens.masked_fill(
            reversed_to_regress_tokens == self.llama_tokenizer.pad_token_id, -100
        ).masked_fill(
            reversed_to_regress_tokens == self.llama_tokenizer.eos_token_id, -100
        ).roll(shifts=-1, dims=1)

        # Calculate loss with separate NTP and register losses
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=reversed_inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                return_dict=True,
                labels=None,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            del outputs
        
            # Prefix length (bos + speech) - register/real indices are relative to after this
            prefix_len = reversed_to_regress_tokens_atts.shape[1]
            vocab_size = logits.shape[-1]
            
            # Calculate NTP loss
            loss_ntp = F.cross_entropy(
                logits[:, :prefix_len, :].contiguous().view(-1, vocab_size),
                targets[:, :prefix_len].contiguous().view(-1),
                ignore_index=-100,
            )
            
            # Calculate audio loss
            audio_preds = last_hidden_state[:, reversed_to_regress_tokens_atts.shape[1]+reversed_p_after_atts.shape[1]:-(reversed_p_before_atts.shape[1]+bos_embeds.shape[1]), :]
            audio_targets = audio_embeds.roll(shifts=-1, dims=1)
            audio_preds[:, -1] = 0.0
            audio_targets[:, -1] = 0.0
            loss_reg = F.mse_loss(
                audio_preds.contiguous().view(-1, audio_preds.shape[-1]),
                audio_targets.contiguous().view(-1, audio_targets.shape[-1]),
            )

        loss = loss_ntp + self.alpha * loss_reg

        if verbose:
            # NTP accuracy (includes prefix→x1 prediction)
            ntp_preds = logits[:, :prefix_len, :].argmax(dim=-1)
            ntp_mask = (targets[:, :prefix_len] != -100)
            ntp_correct = (ntp_preds[ntp_mask] == targets[:, :prefix_len][ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()

            return {
                "loss": loss, 
                "loss_ntp": loss_ntp, 
                "loss_reg": loss_reg, 
                "ntp_correct": ntp_correct, 
                "ntp_total": ntp_total
            }

        return {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}


class MutorSALMONN(SALMONN):
    def __init__(self, min_offset=1, max_offset=4, alpha=0.1, alpha_decay_temperature=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.alpha = alpha
        self.alpha_decay_temperature = alpha_decay_temperature

        # Add register token to the tokenizer
        self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["<reg>"]})
        self.llama_tokenizer.register_token_id = self.llama_tokenizer.convert_tokens_to_ids("<reg>")
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
    
    @classmethod
    def from_config(cls, config):
        # Get MuToR-specific params
        min_offset = config.get("mutor_min_offset", 2)
        max_offset = config.get("mutor_max_offset", 4)
        alpha = config.get("mutor_alpha", 0.1)
        alpha_decay_temperature = config.get("mutor_alpha_decay_temperature", 0.0)
        
        # Get all base SALMONN params
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            min_offset=min_offset,
            max_offset=max_offset,
            alpha=alpha,
            alpha_decay_temperature=alpha_decay_temperature,
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load MutorSALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
    
    def forward(self, samples, verbose=False):
        if not self.training:
            return super().forward(samples, verbose)

        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        ### Add register token within the text ###
        
        batch_size = to_regress_tokens.input_ids.shape[0]
        device = to_regress_tokens.input_ids.device
        double_sequence_len = to_regress_tokens.input_ids.shape[1] * 2

        # Sample offset d uniformly (offset = d - 1 in the code)
        offset = np.random.randint(self.min_offset, self.max_offset + 1)           

        input_ids = []
        attention_mask_4d = []
        reg_token_indices_batch = []
        real_token_indices_batch = []

        for i in range(batch_size):
            answer_part_tensor = to_regress_tokens.input_ids[i]
            attention_mask = to_regress_tokens.attention_mask[i]

            # Create register tokens (same ID for all)
            reg_tokens = torch.full_like(answer_part_tensor, self.llama_tokenizer.register_token_id)
            
            # INTERLEAVE: Stack registers and answer tokens, then flatten
            # Result: [x1, r, x2, r, x3, r, ...]
            interleaved_answer = torch.stack([answer_part_tensor, reg_tokens], dim=1).flatten(0)
            double_attention_mask = attention_mask.repeat_interleave(2)

            # Track indices of register tokens and real tokens
            reg_token_indices = torch.arange(1, len(interleaved_answer), 2, device=device)
            all_indices = torch.arange(double_sequence_len, device=device)
            real_token_mask = torch.ones(double_sequence_len, dtype=bool, device=device)
            real_token_mask[reg_token_indices] = False
            real_token_indices = all_indices[real_token_mask]

            # Create custom attention for the interleaved sequences
            mask_auxiliary = torch.ones([double_sequence_len, double_sequence_len], dtype=speech_embeds.dtype, device=device)
            mask_auxiliary = torch.tril(mask_auxiliary, diagonal=-1) * double_attention_mask.unsqueeze(1)

            # Register tokens cannot attend to other register tokens
            mask_auxiliary[:, ~real_token_mask] = 0.0
            # Restore the diagonal of the mask
            mask_auxiliary.diagonal(dim1=0, dim2=1).copy_(double_attention_mask)

            input_ids.append(interleaved_answer)
            attention_mask_4d.append(mask_auxiliary)
            reg_token_indices_batch.append(reg_token_indices)
            real_token_indices_batch.append(real_token_indices)

        # Update batch with interleaved sequences
        input_ids = torch.stack(input_ids)  # [B, new_seq_len]
        attention_mask_4d = torch.stack(attention_mask_4d).unsqueeze(1)  # [B, 1, new_seq_len, new_seq_len]
        reg_token_indices_batch = torch.stack(reg_token_indices_batch)
        real_token_indices_batch = torch.stack(real_token_indices_batch)

        to_regress_embeds = self.llama_model.model.embed_tokens(input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(input_ids)

        ### Construct labels accordingly to the interleaved sequences ###

        targets = input_ids.clone().roll(-2, dims=1)
        to_regress_position_ids = torch.arange(input_ids.shape[1] // 2, device=input_ids.device).repeat_interleave(2).unsqueeze(0).repeat(batch_size, 1)

        for i in range(batch_size):
            reg_indices = reg_token_indices_batch[i]

            # Compute target indices for each register
            j_tensor = torch.arange(len(reg_indices), device=device)
            target_indices = reg_indices - 1 + 2 * offset  # Target is offset steps ahead
            reg_positions = j_tensor + offset - 1  # Position IDs for registers

            # Mask: target index must be in bounds
            valid_targets_mask = target_indices < double_sequence_len

            # Assign labels: register predicts the token at target_indices
            targets[i, reg_indices[valid_targets_mask]] = input_ids[i, target_indices[valid_targets_mask]]
            targets[i, reg_indices[~valid_targets_mask]] = -100  # Mask out-of-bounds

            # Add position IDs for registers
            last_valid_pos = to_regress_position_ids.shape[1] // 2 - 1
            to_regress_position_ids[i, reg_indices[valid_targets_mask]] = reg_positions[valid_targets_mask]
            to_regress_position_ids[i, reg_indices[~valid_targets_mask]] = last_valid_pos
        
        targets[targets == self.llama_tokenizer.pad_token_id] = -100 # Mask pad tokens
        targets[:, -2:] = -100 # Mask last two tokens (due to the shift of the interleaved sequences)

        ### Construct inputs + targets for LLM ###

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)

        # Construct inputs for LLM
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)

        # Construct attention mask for the interleaved sequences
        total_len = bos_embeds.shape[1] + speech_embeds.shape[1] + to_regress_embeds.shape[1]
        start_regress = bos_embeds.shape[1] + speech_embeds.shape[1]

        # Create base 4D mask with 1s (attend) - will use causal masking from LLaMA
        mask_4d = torch.ones([batch_size, 1, total_len, total_len], dtype=speech_embeds.dtype, device=device)
        mask_4d[:, 0, start_regress:, start_regress:] = attention_mask_4d[:, 0, :, :]
        mask_4d = torch.tril(mask_4d, diagonal=0)

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min))

        # Position IDs for registers
        position_ids = torch.cat([
            torch.arange(start_regress, device=device).unsqueeze(0).repeat(batch_size, 1), 
            to_regress_position_ids + start_regress
        ], dim=1)

        # Calculate loss with separate NTP and register losses
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                return_dict=True,
                labels=None,
            )
            logits = outputs.logits
        
            # Prefix length (bos + speech) - register/real indices are relative to after this
            prefix_len = bos_embeds.shape[1] + speech_embeds.shape[1]
            vocab_size = logits.shape[-1]
            seq_len = targets.shape[1]
            
            flat_logits = logits[:, prefix_len:, :].reshape(-1, vocab_size)
            flat_targets = targets.view(-1)

            # Add batch offsets to convert per-sample indices to flat indices
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * seq_len
            flat_real_indices = (real_token_indices_batch + batch_offsets).view(-1)
            flat_reg_indices = (reg_token_indices_batch + batch_offsets).view(-1)

            # Get logits and targets for real and register tokens
            real_logits = flat_logits[flat_real_indices]
            reg_logits = flat_logits[flat_reg_indices]
            real_targets = flat_targets[flat_real_indices]
            reg_targets = flat_targets[flat_reg_indices]

            # Include prefix → x1 prediction (the audio-to-text bridge)
            num_real = real_token_indices_batch.shape[1]
            prefix_last_logit = logits[:, prefix_len - 1, :].unsqueeze(1)
            first_token_target = input_ids[:, 0].clone().unsqueeze(1)
            first_token_target[first_token_target == self.llama_tokenizer.pad_token_id] = -100

            all_ntp_logits = torch.cat([prefix_last_logit, real_logits.view(batch_size, num_real, -1)], dim=1).reshape(-1, vocab_size)
            all_ntp_targets = torch.cat([first_token_target, real_targets.view(batch_size, num_real)], dim=1).reshape(-1)

            # Compute loss for NTP and register tokens
            loss_ntp = F.cross_entropy(all_ntp_logits, all_ntp_targets)
            loss_reg = F.cross_entropy(reg_logits, reg_targets)

            if self.alpha_decay_temperature > 0.0:
                alpha = self.alpha * torch.exp(-torch.tensor(max(0, offset - 1), dtype=torch.float)/self.alpha_decay_temperature)
            else:
                alpha = self.alpha

            # Combined loss (can be weighted if needed)
            loss = (1 - alpha) * loss_ntp + alpha * loss_reg

        if verbose:
            # NTP accuracy (includes prefix→x1 prediction)
            ntp_preds = all_ntp_logits.argmax(dim=-1)
            ntp_mask = (all_ntp_targets != -100)
            ntp_correct = (ntp_preds[ntp_mask] == all_ntp_targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()

            # Register accuracy
            reg_preds = reg_logits.argmax(dim=-1)
            reg_mask = (reg_targets != -100)
            reg_correct = (reg_preds[reg_mask] == reg_targets[reg_mask]).float().sum()
            reg_total = reg_mask.sum().item()

            # Overall accuracy
            correct = ntp_correct + reg_correct
            total = ntp_total + reg_total

            return {
                "loss": loss,
                "loss_ntp": loss_ntp,
                "loss_reg": loss_reg,
                "ntp_correct": ntp_correct,
                "ntp_total": ntp_total,
                "reg_correct": reg_correct,
                "reg_total": reg_total,
                "correct": correct,
                "total": total,
            }

        return {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}


class MutorBERTSummarySALMONN(MutorSALMONN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.chapter_tokenizer = AutoTokenizer.from_pretrained(_bert_model_name)
        self.chapter_encoder = AutoModel.from_pretrained(_bert_model_name)
        self.chapter_encoder.eval()
        for param in self.chapter_encoder.parameters():
            param.requires_grad = False
        self.chapter_encoder.to(self.device)

        self.chapter_proj = nn.Linear(
            self.llama_model.config.hidden_size, #256
            384, # BERT embedding size
        )

    def forward(self, samples, verbose=False):
        if not self.training:
            return super(MutorSALMONN, self).forward(samples, verbose)

        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]

        # Extract chapter summary from text & add register tokens before each chapter
        text_with_registers = []
        all_chapter_texts = []
        chapter_names = ["SUMMARY", "CAPTION", "REASONING", "CONCLUSION"]
        for t in text:
            tmp_text = []
            for chapter_name in chapter_names:
                match = re.search(rf"<{chapter_name}>(.*?)</{chapter_name}>", t, re.DOTALL)
                chapter_text = match.group(1).strip()
                tmp_text.append("<reg>")
                tmp_text.append(f"<{chapter_name}>")
                tmp_text.append(chapter_text)
                tmp_text.append(f"</{chapter_name}>")
                all_chapter_texts.append(chapter_text)
            text_with_registers.append("".join(tmp_text))

        # Batch BERT encoding for all chapters at once
        encoded = self.chapter_tokenizer(
            all_chapter_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            bert_out = self.chapter_encoder(**encoded)

            # Get the token embeddings and the attention mask
            token_embeddings = bert_out.last_hidden_state
            attention_mask = encoded["attention_mask"]

            # Expand the mask to match embeddings shape
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            # Sum embeddings, ignoring padded tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # Divide by the number of non-padded tokens
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            chapter_embeds = sum_embeddings / sum_mask
        del encoded, bert_out
        num_chapters = len(chapter_names)
        chapter_embeds = chapter_embeds.view(len(text), num_chapters, -1)

        # Tokenize text with registers
        to_regress_tokens = self.llama_tokenizer(
            text_with_registers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        batch_size = to_regress_tokens.input_ids.shape[0]
        device = to_regress_tokens.input_ids.device

        # Get ids for register tokens
        reg_token_ids = self.llama_tokenizer.convert_tokens_to_ids("<reg>")
        reg_token_indices = (to_regress_tokens.input_ids == reg_token_ids).nonzero(as_tuple=True)

        attention_mask = to_regress_tokens.attention_mask
        attention_mask[reg_token_indices] = 0.0

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)

        ### Construct inputs + targets for LLM ###

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)

        # Construct inputs for LLM
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)

        # Construct attention mask for the interleaved sequences
        total_len = bos_embeds.shape[1] + speech_embeds.shape[1] + to_regress_embeds.shape[1]
        start_regress = bos_embeds.shape[1] + speech_embeds.shape[1]

        # Create base 4D causal mask
        mask_4d = torch.ones([batch_size, 1, total_len, total_len], dtype=speech_embeds.dtype, device=device)
        mask_4d = torch.tril(mask_4d, diagonal=0)

        # Apply 2D attention mask: zero out columns where attention_mask == 0
        mask_4d[:, 0, :, start_regress:] *= attention_mask.unsqueeze(1).to(mask_4d.dtype)
        reg_positions_in_full = reg_token_indices[1] + start_regress
        mask_4d[reg_token_indices[0], 0, reg_positions_in_full, reg_positions_in_full] = 1.0

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min))

        # Position IDs for registers
        position_ids = torch.cat([
            torch.arange(start_regress, device=device).unsqueeze(0).repeat(batch_size, 1), 
            torch.cumsum(attention_mask, dim=1) + start_regress - 1
        ], dim=1)

        # Construct targets for the sequence without registers
        targets = torch.full_like(to_regress_tokens.input_ids, -100)
        for i in range(batch_size):
            valid_mask = attention_mask[i].bool()
            valid_tokens = to_regress_tokens.input_ids[i, valid_mask]
            targets[i, valid_mask] = torch.cat([valid_tokens[1:], torch.tensor([-100], device=device)])

        # Calculate loss with separate NTP and register losses
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                return_dict=True,
                labels=None,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            del outputs

        prefix_len = bos_embeds.shape[1] + speech_embeds.shape[1]
        vocab_size = logits.shape[-1]

        shift_logits = logits[:, prefix_len:-1, :].reshape(-1, vocab_size)
        shift_targets = targets[:, :-1].reshape(-1)
        loss_ntp = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)

        # Register embedding loss
        reg_batch_idx, reg_seq_idx = reg_token_indices
        reg_hidden = last_hidden_state[reg_batch_idx, prefix_len + reg_seq_idx]
        reg_projected = self.chapter_proj(reg_hidden)
        _, counts = torch.unique(reg_batch_idx, return_counts=True)
        existing_chapters_mask = torch.arange(num_chapters, device=counts.device) < counts.unsqueeze(1)
        reg_aligned = reg_projected.new_zeros(batch_size, num_chapters, reg_projected.shape[-1])
        reg_aligned[existing_chapters_mask] = reg_projected
        cos_sim = F.cosine_similarity(reg_aligned, chapter_embeds, dim=-1)
        cos_sim = cos_sim.masked_fill(~existing_chapters_mask, 1.0)
        loss_reg = (1 - cos_sim).sum() / existing_chapters_mask.sum().clamp(min=1e-9)

        loss = loss_ntp + self.alpha * loss_reg

        if verbose:
            # NTP accuracy (includes prefix→x1 prediction)
            ntp_preds = shift_logits.argmax(dim=-1)
            ntp_mask = (shift_targets != -100)
            ntp_correct = (ntp_preds[ntp_mask] == shift_targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()

            return {
                "loss": loss,
                "loss_ntp": loss_ntp,
                "loss_reg": loss_reg,
                "ntp_correct": ntp_correct,
                "ntp_total": ntp_total,
            }

        return {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}


class MutorBERTConclusionSALMONN(MutorSALMONN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.conclusion_tokenizer = AutoTokenizer.from_pretrained(_bert_model_name)
        self.conclusion_encoder = AutoModel.from_pretrained(_bert_model_name)
        self.conclusion_encoder.eval()
        for param in self.conclusion_encoder.parameters():
            param.requires_grad = False

        self.conclusion_proj = nn.Linear(
            self.llama_model.config.hidden_size, #256
            384, # BERT embedding size
        )

    def forward(self, samples, verbose=False, output_attentions=False):
        if not self.training:
            return super(MutorSALMONN, self).forward(samples, verbose)

        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]

        # Extract chapter summary from text & add register tokens before each chapter
        all_conclusion_texts = []
        for i, t in enumerate(text):
            match = re.search(rf"<CONCLUSION>(.*?)</CONCLUSION>", t, re.DOTALL)
            conclusion_text = match.group(1).strip()
            all_conclusion_texts.append(conclusion_text)
            text[i] = "<reg>" + t

        # Batch BERT encoding for all chapters at once
        encoded = self.conclusion_tokenizer(
            all_conclusion_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            bert_out = self.conclusion_encoder(**encoded)
            conclusion_embeds = F.normalize(bert_out.last_hidden_state[:, 0, :], p=2, dim=1)

        # Tokenize text with registers
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        batch_size = to_regress_tokens.input_ids.shape[0]
        device = to_regress_tokens.input_ids.device

        # Set attention mask for registers to 0
        attention_mask = to_regress_tokens.attention_mask
        attention_mask[:, 0] = 0.0

        # Get token embeddings
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)

        ### Construct inputs + targets for LLM ###

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)

        # Construct inputs for LLM
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)

        # Construct attention mask for the interleaved sequences
        total_len = bos_embeds.shape[1] + speech_embeds.shape[1] + to_regress_embeds.shape[1]
        start_regress = bos_embeds.shape[1] + speech_embeds.shape[1]

        # Create base 4D causal mask
        mask_4d = torch.ones([batch_size, 1, total_len, total_len], dtype=speech_embeds.dtype, device=device)
        mask_4d = torch.tril(mask_4d, diagonal=0)

        # Apply 2D attention mask: zero out columns where attention_mask == 0
        mask_4d[:, 0, :, start_regress:] *= attention_mask.unsqueeze(1).to(mask_4d.dtype)
        mask_4d[:, 0, start_regress, start_regress] = 1.0

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min))

        # Position IDs
        position_ids = torch.cat([
            torch.arange(start_regress, device=device).unsqueeze(0).repeat(batch_size, 1), 
            torch.cumsum(attention_mask, dim=1) + start_regress - 1
        ], dim=1)

        # Construct targets for the sequence without registers
        targets = torch.full_like(to_regress_tokens.input_ids, -100)
        for i in range(batch_size):
            valid_mask = attention_mask[i].bool()
            valid_tokens = to_regress_tokens.input_ids[i, valid_mask]
            targets[i, valid_mask] = torch.cat([valid_tokens[1:], torch.tensor([-100], device=device)])

        # Calculate loss with separate NTP and register losses
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                return_dict=True,
                labels=None,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            attentions = outputs.attentions if output_attentions else None
            del outputs

            vocab_size = logits.shape[-1]
            logits = logits[:, start_regress-1:, :].reshape(-1, vocab_size)
            targets = torch.cat([torch.full((batch_size, 1), 529, device=device, dtype=torch.long), targets], dim=1).reshape(-1)
            loss_ntp = F.cross_entropy(logits, targets, ignore_index=-100)

            # Register embedding loss
            reg_hidden = last_hidden_state[:, start_regress, :]
            reg_projected = self.conclusion_proj(reg_hidden)
            cos_sim = F.cosine_similarity(reg_projected, conclusion_embeds, dim=-1)
            loss_reg = (1 - cos_sim).mean()

            loss = loss_ntp + self.alpha * loss_reg

        if verbose:
            # NTP accuracy (includes prefix→x1 prediction)
            ntp_preds = logits.argmax(dim=-1)
            ntp_mask = (targets != -100)
            ntp_correct = (ntp_preds[ntp_mask] == targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()

            out = {
                "loss": loss,
                "loss_ntp": loss_ntp,
                "loss_reg": loss_reg,
                "ntp_correct": ntp_correct,
                "ntp_total": ntp_total,
            }
        else:
            out = {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}

        if output_attentions:
            out["attentions"] = torch.stack(attentions, dim=0)
            out["attentions_meta"] = {
                "to_regress_tokens": to_regress_tokens.input_ids,
                "start_regress": start_regress,
                "speech_len": speech_embeds.shape[1],
                "mask_4d": mask_4d.bool(),
            }
        
        return out


class MutorBERTTripletLossSALMONN(MutorSALMONN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.conclusion_tokenizer = AutoTokenizer.from_pretrained(_bert_model_name)
        self.conclusion_encoder = AutoModel.from_pretrained(_bert_model_name)
        self.conclusion_encoder.eval()
        for param in self.conclusion_encoder.parameters():
            param.requires_grad = False

        self.conclusion_proj = nn.Linear(
            self.llama_model.config.hidden_size, #256
            384, # BERT embedding size
        )

    def forward(self, samples, verbose=False):
        if not self.training:
            return super(MutorSALMONN, self).forward(samples, verbose)

        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

            # For reasoning tasks, concatenate question + prompt
            if "question" in samples and any(samples["question"]):
                if not self.multi_prompt:
                    prompt = [prompt] * len(samples["question"])
                    self.multi_prompt = True
                prompt = [q + " " + p for p, q in zip(prompt, samples["question"])]

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts (includes question for reasoning tasks)
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM (use answer for reasoning tasks, text for ASR)
        if "answer" in samples and any(samples["answer"]):
            text = [t + self.end_sym for t in samples["answer"]]
        else:
            text = [t + self.end_sym for t in samples["text"]]

        # Extract chapter summary from text & add register tokens before each chapter
        option_embeds = torch.zeros(len(text), 4, 384, device=self.device)
        num_options_per_sample = []
        for i, t in enumerate(text):
            # Add register tokens before the conclusion text
            t = "<reg>" + t

            # Extract the correct option text
            match = re.search(rf"<CONCLUSION>(.*?)</CONCLUSION>", t, re.DOTALL)
            correct_option_text = re.sub(r"^\([A-Za-z]\)\s*", "", match.group(1).strip()).replace(".", "").strip()

            # Extract the options text (handle missing delimiter, e.g. different dataset formats)
            _parts = samples["question"][i].split("Choose one among the following options:")
            question_text = _parts[1] if len(_parts) > 1 else ""
            options_text = [
                re.sub(r"^\([A-Za-z]\)\s*", "", line.strip()).replace(".", "").strip()
                for line in question_text.split("\n")
                if line.strip()
            ]

            # Remove the correct option from the options text
            options_text = [option for option in options_text if option != correct_option_text][:3]

            # Batch BERT encoding for the correct option and options texts
            encoded = self.conclusion_tokenizer(
                [correct_option_text] + options_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                bert_out = self.conclusion_encoder(**encoded)
                option_embeds[i, :len(options_text)+1, :] = F.normalize(bert_out.last_hidden_state[:, 0, :], p=2, dim=1)
            num_options_per_sample.append(len(options_text) + 1)

        # Tokenize text with registers
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)

        batch_size = to_regress_tokens.input_ids.shape[0]
        device = to_regress_tokens.input_ids.device

        # Set attention mask for registers to 0
        attention_mask = to_regress_tokens.attention_mask
        attention_mask[:, 0] = 0.0

        # Get token embeddings
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)

        ### Construct inputs + targets for LLM ###

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)

        # Construct inputs for LLM
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)

        # Construct attention mask for the interleaved sequences
        total_len = bos_embeds.shape[1] + speech_embeds.shape[1] + to_regress_embeds.shape[1]
        start_regress = bos_embeds.shape[1] + speech_embeds.shape[1]

        # Create base 4D causal mask
        mask_4d = torch.ones([batch_size, 1, total_len, total_len], dtype=speech_embeds.dtype, device=device)
        mask_4d = torch.tril(mask_4d, diagonal=0)

        # Apply 2D attention mask: zero out columns where attention_mask == 0
        mask_4d[:, 0, :, start_regress:] *= attention_mask.unsqueeze(1).to(mask_4d.dtype)
        mask_4d[:, 0, start_regress, start_regress] = 1.0

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min))

        # Position IDs
        position_ids = torch.cat([
            torch.arange(start_regress, device=device).unsqueeze(0).repeat(batch_size, 1), 
            torch.cumsum(attention_mask, dim=1) + start_regress - 1
        ], dim=1)

        # Construct targets for the sequence without registers
        targets = torch.full_like(to_regress_tokens.input_ids, -100)
        for i in range(batch_size):
            valid_mask = attention_mask[i].bool()
            valid_tokens = to_regress_tokens.input_ids[i, valid_mask]
            targets[i, valid_mask] = torch.cat([valid_tokens[1:], torch.tensor([-100], device=device)])

        # Calculate loss with separate NTP and register losses
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                return_dict=True,
                labels=None,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            del outputs

            vocab_size = logits.shape[-1]
            logits = logits[:, start_regress-1:, :].reshape(-1, vocab_size)
            targets = torch.cat([torch.full((batch_size, 1), 529, device=device, dtype=torch.long), targets], dim=1).reshape(-1)
            loss_ntp = F.cross_entropy(logits, targets, ignore_index=-100)

            # Register contrastive loss
            reg_hidden = last_hidden_state[:, start_regress, :]
            reg_projected = F.normalize(self.conclusion_proj(reg_hidden), p=2, dim=-1)
            pos_embed = option_embeds[:, 0, :]
            sim_pos = (reg_projected * pos_embed).sum(dim=-1)
            tau = 0.07
            loss_reg = 0.0
            for i in range(batch_size):
                n = num_options_per_sample[i]
                neg_embeds = option_embeds[i, 1:n, :]
                sim_neg = (reg_projected[i : i + 1] * neg_embeds).sum(dim=-1)
                logits_reg = torch.cat([sim_pos[i : i + 1] / tau, sim_neg / tau]).unsqueeze(0)
                loss_reg = loss_reg + F.cross_entropy(logits_reg, torch.zeros(1, dtype=torch.long, device=device))
            loss_reg = loss_reg / batch_size

            loss = loss_ntp + self.alpha * loss_reg

        if verbose:
            # NTP accuracy (includes prefix→x1 prediction)
            ntp_preds = logits.argmax(dim=-1)
            ntp_mask = (targets != -100)
            ntp_correct = (ntp_preds[ntp_mask] == targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()

            return {
                "loss": loss,
                "loss_ntp": loss_ntp,
                "loss_reg": loss_reg,
                "ntp_correct": ntp_correct,
                "ntp_total": ntp_total,
            }

        return {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}

