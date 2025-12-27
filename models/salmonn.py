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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList
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

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
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
        # Use float16 on GPU for efficiency, float32 on CPU for stability
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=model_dtype,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=model_dtype,
            )
        logging.info(f'Model dtype: {model_dtype}')

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
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
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

    def forward(self, samples, verbose=False):
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

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
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


class MutorSALMONN(SALMONN):
    def __init__(self, min_offset=1, max_offset=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_offset = min_offset
        self.max_offset = max_offset

        # Add register token to the tokenizer
        self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["<reg>"]})
        self.llama_tokenizer.register_token_id = self.llama_tokenizer.convert_tokens_to_ids("<reg>")
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
    
    def forward(self, samples, verbose=False):
        if not self.training:
            return super().forward(samples, verbose)

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

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)
        
        # prepare inputs for LLM
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
            # Result: [r, x1, r, x2, r, x3, ...]
            interleaved_answer = torch.stack([reg_tokens, answer_part_tensor], dim=1).flatten(0)
            double_attention_mask = attention_mask.repeat_interleave(2)
            double_sequence_len = double_attention_mask.shape[0]

            # Track indices of register tokens and real tokens
            reg_token_indices = torch.arange(0, len(interleaved_answer), 2, device=device)
            total_len = interleaved_answer.size(0)
            all_indices = torch.arange(total_len, device=device)
            real_token_mask = torch.ones(total_len, dtype=bool, device=device)
            real_token_mask[reg_token_indices] = False
            real_token_indices = all_indices[real_token_mask]

            # Create custom attention for the interleaved sequences
            mask_auxiliary = torch.ones([double_sequence_len, double_sequence_len], dtype=speech_embeds.dtype, device=device)
            mask_auxiliary = torch.tril(mask_auxiliary, diagonal=-1) * double_attention_mask.unsqueeze(1)

            # Register tokens cannot attend to other register tokens
            mask_auxiliary[:, ~real_token_mask] = 0.0
            # Restore the diagonal of the mask
            mask_auxiliary.diagonal(dim1=0, dim2=1).fill_(1.0)

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
        targets[targets == self.llama_tokenizer.pad_token_id] = -100
        targets[:, -1] = -100
        to_regress_position_ids = torch.arange(input_ids.shape[1] // 2, device=input_ids.device).repeat_interleave(2).unsqueeze(0).repeat(batch_size, 1)

        reg_position_ids_batch = []
        for i in range(batch_size):
            reg_indices = reg_token_indices_batch[i]

            # Compute target indices for each register
            j_tensor = torch.arange(len(reg_indices), device=device)
            target_indices = reg_indices + 1 + 2 * offset  # Target is offset steps ahead
            reg_positions = j_tensor + offset - 1  # Position IDs for registers

            # Mask: target index must be in bounds
            valid_targets_mask = target_indices < double_sequence_len

            # Assign labels: register predicts the token at target_indices
            targets[i, reg_indices[valid_targets_mask]] = input_ids[i, target_indices[valid_targets_mask]]
            targets[i, reg_indices[~valid_targets_mask]] = -100  # Mask out-of-bounds

            # Store position IDs for registers
            last_valid_pos = len(real_token_indices_batch[i]) - 1
            reg_position_ids = torch.where(
                valid_targets_mask,
                reg_positions,
                torch.full_like(reg_positions, last_valid_pos)
            )
            reg_position_ids_batch.append(reg_position_ids)

            # Update position IDs for registers
            to_regress_position_ids[i, reg_token_indices_batch[i]] = reg_position_ids_batch[i]

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

        # Convert to additive mask format (1 → 0.0, 0 → -inf)
        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(mask_4d > 0.5, torch.tensor(torch.finfo(mask_4d.dtype).min))

        # Position IDs for registers
        position_ids = torch.cat([
            torch.arange(start_regress, device=device).unsqueeze(0).repeat(batch_size, 1), 
            to_regress_position_ids + start_regress
        ], dim=1)

        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long,
                device=device,
            ).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
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