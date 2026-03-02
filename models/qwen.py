import re
import logging
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
    AutoTokenizer,
    AutoModel,
)


logger = logging.getLogger(__name__)

_SALMONN_AUDIO_TAG = "<Speech><SpeechHere></Speech>"
_QWEN_AUDIO_TAG = "<|audio_bos|><|AUDIO|><|audio_eos|>"


class Qwen25Omni(nn.Module):
    """
    Qwen2.5-Omni wrapper supporting both zero-shot inference and NTP
    fine-tuning on audio benchmarks (MMAU, MMAR, etc.).

    Uses Qwen2_5OmniThinkerForConditionalGeneration (text-only, no talker).
    Supports LoRA on the text backbone with frozen audio/visual encoders.
    For zero-shot evaluation, instantiate without LoRA and call generate().

    Subclasses can pass ``additional_special_tokens`` to inject extra tokens
    (e.g. <reg>) *before* LoRA is applied.
    """

    def __init__(
        self,
        qwen25_omni_path: str,
        max_txt_len: int = 400,
        freeze_audio_tower: bool = True,
        freeze_visual: bool = True,
        lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        additional_special_tokens: list = None,
    ):
        super().__init__()
        self.max_txt_len = max_txt_len
        self.lora = lora

        # bfloat16 on GPU; float32 on CPU for compatibility (bfloat16 on CPU is slow)
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            qwen25_omni_path,
            torch_dtype=model_dtype,
        )
        logger.info("Qwen thinker dtype: %s", model_dtype)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(qwen25_omni_path)
        self.tokenizer = self.processor.tokenizer

        if additional_special_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )
            self.thinker.resize_token_embeddings(len(self.tokenizer))

        if freeze_audio_tower:
            for param in self.thinker.audio_tower.parameters():
                param.requires_grad = False
            self.thinker.audio_tower.eval()
            logger.info("Froze audio_tower")

        if freeze_visual:
            for param in self.thinker.visual.parameters():
                param.requires_grad = False
            self.thinker.visual.eval()
            logger.info("Froze visual encoder")

        if lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.thinker.model = get_peft_model(self.thinker.model, lora_cfg)
            # lm_head is outside the PEFT wrapper; freeze it so only LoRA params are trainable
            if hasattr(self.thinker, "lm_head") and self.thinker.lm_head is not None:
                for param in self.thinker.lm_head.parameters():
                    param.requires_grad = False
                logger.info("Froze lm_head (LoRA mode)")
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logger.info(
                "Applied LoRA (r=%d, alpha=%d) to thinker.model | trainable: %s / %s (%.2f%%)",
                lora_rank,
                lora_alpha,
                f"{trainable:,}",
                f"{total:,}",
                100.0 * trainable / total if total > 0 else 0,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_embed_tokens(self):
        return self.thinker.get_input_embeddings()

    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def _encode_audio(self, input_features, feature_attention_mask):
        """
        Encode audio through the thinker's audio tower.

        Returns:
            audio_features_flat: (total_audio_tokens, hidden_dim) — flat across batch
            audio_output_lengths: (batch_size,) — per-sample token counts
        """
        mel_lengths = feature_attention_mask.sum(-1)
        _, audio_output_lengths = self.thinker.audio_tower._get_feat_extract_output_lengths(
            mel_lengths
        )
        audio_outputs = self.thinker.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
        )
        # transformers 5.x returns tensor directly; older versions return object with .last_hidden_state
        audio_hidden = (
            audio_outputs.last_hidden_state
            if hasattr(audio_outputs, "last_hidden_state")
            else audio_outputs
        )
        return audio_hidden, audio_output_lengths

    def forward(self, samples, verbose=False):
        """
        Training forward with standard NTP loss.

        samples dict keys:
            input_features:         (B, n_mels, mel_len)
            feature_attention_mask: (B, mel_len)
            text / answer:          list[str]
            question:               list[str] (optional)
            task:                   list[str]
        """
        if not self.training:
            return self._inference_forward(samples, verbose)

        device = self.device
        input_features = samples["input_features"].to(device)
        feature_attention_mask = samples["feature_attention_mask"].to(device)

        with self.maybe_autocast():
            audio_features_flat, audio_output_lengths = self._encode_audio(
                input_features, feature_attention_mask
            )

        if "answer" in samples and any(samples["answer"]):
            raw_texts = samples["answer"]
        else:
            raw_texts = samples["text"]

        batch_size = input_features.shape[0]
        questions = samples.get("question", [""] * batch_size)

        prompt_texts = []
        for i in range(batch_size):
            n_audio_tokens = audio_output_lengths[i].item()
            audio_placeholder = "<|AUDIO|>" * n_audio_tokens
            prompt = f"<|audio_bos|>{audio_placeholder}<|audio_eos|>"
            if questions[i]:
                prompt += f" {questions[i]}"
            prompt_texts.append(prompt)

        prompt_tokens = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
        ).to(device)

        target_tokens = self.tokenizer(
            raw_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(device)

        target_ids = target_tokens.input_ids
        target_attn = target_tokens.attention_mask

        embed_tokens = self._get_embed_tokens()
        prompt_embeds = embed_tokens(prompt_tokens.input_ids)
        target_embeds = embed_tokens(target_ids)

        audio_token_id = self.thinker.config.audio_token_id
        audio_mask = prompt_tokens.input_ids == audio_token_id
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(
            audio_mask_3d, audio_features_flat.to(prompt_embeds.dtype)
        )

        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        prompt_len = prompt_embeds.shape[1]
        total_len = prompt_len + target_embeds.shape[1]

        mask_4d = torch.ones(
            [batch_size, 1, total_len, total_len],
            dtype=inputs_embeds.dtype,
            device=device,
        )
        mask_4d = torch.tril(mask_4d, diagonal=0)

        mask_4d[:, 0, :, prompt_len:] *= target_attn.unsqueeze(1).to(mask_4d.dtype)
        prompt_attn = prompt_tokens.attention_mask
        mask_4d[:, 0, :, :prompt_len] *= prompt_attn.unsqueeze(1).to(mask_4d.dtype)

        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(
            mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min)
        )

        # 3D position IDs for TMRoPE: (3, B, total_len)
        prompt_position_ids = torch.cumsum(prompt_attn, dim=1) - 1
        prompt_position_ids = prompt_position_ids.clamp(min=0)
        last_valid_prompt_pos = (prompt_attn.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
        target_position_ids = torch.cumsum(target_attn, dim=1) + last_valid_prompt_pos
        position_ids_1d = torch.cat(
            [prompt_position_ids, target_position_ids], dim=1
        )
        position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)

        targets = torch.full_like(target_ids, -100)
        for i in range(batch_size):
            valid_mask = target_attn[i].bool()
            valid_tokens = target_ids[i, valid_mask]
            targets[i, valid_mask] = torch.cat(
                [valid_tokens[1:], torch.tensor([-100], device=device)]
            )

        with self.maybe_autocast():
            text_outputs = self.thinker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            logits = self.thinker.lm_head(text_outputs.last_hidden_state)
            del text_outputs

            vocab_size = logits.shape[-1]
            logits_for_loss = logits[:, prompt_len - 1:, :].reshape(-1, vocab_size)

            first_target = target_ids[:, 0:1].clone()
            first_target[target_attn[:, 0:1] == 0] = -100
            full_targets = torch.cat([first_target, targets], dim=1).reshape(-1)

            loss = F.cross_entropy(
                logits_for_loss, full_targets, ignore_index=-100
            )

        out = {"loss": loss, "loss_ntp": loss}

        if verbose:
            ntp_preds = logits_for_loss.argmax(dim=-1)
            ntp_mask = full_targets != -100
            ntp_correct = (ntp_preds[ntp_mask] == full_targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()
            out.update({"ntp_correct": ntp_correct, "ntp_total": ntp_total})

        return out

    @torch.no_grad()
    def _inference_forward(self, samples, verbose=False):
        """Inference forward with standard NTP loss."""
        device = self.device
        input_features = samples["input_features"].to(device)
        feature_attention_mask = samples["feature_attention_mask"].to(device)

        if "answer" in samples and any(samples["answer"]):
            raw_texts = samples["answer"]
        else:
            raw_texts = samples["text"]

        batch_size = input_features.shape[0]
        questions = samples.get("question", [""] * batch_size)

        full_texts = []
        for i in range(batch_size):
            prompt = _QWEN_AUDIO_TAG
            if questions[i]:
                prompt += f" {questions[i]}"
            full_texts.append(prompt + " " + raw_texts[i])

        inputs = self.processor(
            text=full_texts,
            audios=None,
            return_tensors="pt",
            padding=True,
        ).to(device)

        labels = inputs.input_ids.clone()
        outputs = self.thinker(
            input_ids=inputs.input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=inputs.attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=False,
        )
        loss = (
            outputs.loss
            if outputs.loss is not None
            else torch.tensor(0.0, device=device)
        )

        out = {
            "loss": loss,
            "loss_ntp": loss,
            "loss_reg": torch.tensor(0.0, device=device),
        }
        if verbose:
            logits = outputs.logits[:, :-1, :]
            preds = logits.argmax(dim=-1)
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            correct = (preds[mask] == shifted_labels[mask]).float().sum()
            total = mask.sum().item()
            out.update({
                "ntp_correct": correct,
                "ntp_total": total,
            })

        return out

    @torch.no_grad()
    def generate(self, samples, generate_cfg, prompts=None):
        """
        Generate text from audio inputs.

        When *prompts* come from a SALMONN-style template (containing
        ``<Speech><SpeechHere></Speech>``), the tag is automatically replaced
        with the Qwen audio placeholder so the same prompt functions can be
        reused across model families.
        """
        device = self.device
        input_features = samples["input_features"].to(device)
        feature_attention_mask = samples["feature_attention_mask"].to(device)
        batch_size = input_features.shape[0]

        questions = samples.get("question", [""] * batch_size)

        if prompts is not None:
            prompt_texts = [
                p.replace(_SALMONN_AUDIO_TAG, _QWEN_AUDIO_TAG)
                for p in prompts
            ]
        else:
            prompt_texts = []
            for i in range(batch_size):
                prompt = _QWEN_AUDIO_TAG
                if questions[i]:
                    prompt += f" {questions[i]}"
                prompt_texts.append(prompt)

        inputs = self.processor(
            text=prompt_texts,
            audios=None,
            return_tensors="pt",
            padding=True,
        ).to(device)

        generate_ids = self.thinker.generate(
            input_ids=inputs.input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=inputs.attention_mask,
            max_new_tokens=generate_cfg.get("max_new_tokens", 300),
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @classmethod
    def from_config(cls, config):
        model = cls(
            qwen25_omni_path=config.get(
                "qwen25_omni_path", "Qwen/Qwen2.5-Omni-7B"
            ),
            max_txt_len=config.get("max_txt_len", 300),
            freeze_audio_tower=config.get("freeze_audio_tower", True),
            freeze_visual=config.get("freeze_visual", True),
            lora=config.get("lora", False),
            lora_rank=config.get("lora_rank", 8),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logger.info("Loading checkpoint from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        return model


class MutorBERTConclusionQwen25Omni(Qwen25Omni):
    """
    Extends Qwen25Omni with the MuToR BERT Conclusion approach: a <reg>
    register token whose hidden state is trained to match a BERT encoding
    of the <CONCLUSION>...</CONCLUSION> text, on top of standard NTP loss.

    Additional components over the base class:
      - conclusion_encoder: frozen BERT (all-MiniLM-L6-v2)
      - conclusion_proj:    Linear(llm_hidden -> 384)
    """

    def __init__(
        self,
        qwen25_omni_path: str,
        alpha: float = 1.0,
        max_txt_len: int = 400,
        freeze_audio_tower: bool = True,
        freeze_visual: bool = True,
        lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bert_conclusion_path: str = None,
    ):
        super().__init__(
            qwen25_omni_path=qwen25_omni_path,
            max_txt_len=max_txt_len,
            freeze_audio_tower=freeze_audio_tower,
            freeze_visual=freeze_visual,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            additional_special_tokens=["<reg>"],
        )
        self.alpha = alpha
        self.reg_token_id = self.tokenizer.convert_tokens_to_ids("<reg>")

        _bert_model_name = bert_conclusion_path or "sentence-transformers/all-MiniLM-L6-v2"
        _local_files_only = bert_conclusion_path is not None
        self.conclusion_tokenizer = AutoTokenizer.from_pretrained(
            _bert_model_name, local_files_only=_local_files_only
        )
        self.conclusion_encoder = AutoModel.from_pretrained(
            _bert_model_name, local_files_only=_local_files_only
        )
        self.conclusion_encoder.eval()
        for param in self.conclusion_encoder.parameters():
            param.requires_grad = False

        llm_hidden_size = self.thinker.config.text_config.hidden_size
        self.conclusion_proj = nn.Linear(llm_hidden_size, 384)

    def forward(self, samples, verbose=False):
        """Training forward with dual NTP + register-BERT loss."""
        if not self.training:
            return self._inference_forward(samples, verbose)

        device = self.device
        input_features = samples["input_features"].to(device)
        feature_attention_mask = samples["feature_attention_mask"].to(device)

        with self.maybe_autocast():
            audio_features_flat, audio_output_lengths = self._encode_audio(
                input_features, feature_attention_mask
            )

        if "answer" in samples and any(samples["answer"]):
            raw_texts = samples["answer"]
        else:
            raw_texts = samples["text"]

        all_conclusion_texts = []
        texts_with_reg = []
        for t in raw_texts:
            match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", t, re.DOTALL)
            conclusion_text = match.group(1).strip()
            all_conclusion_texts.append(conclusion_text)
            texts_with_reg.append("<reg>" + t)

        encoded = self.conclusion_tokenizer(
            all_conclusion_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            bert_out = self.conclusion_encoder(**encoded)
            conclusion_embeds = F.normalize(
                bert_out.last_hidden_state[:, 0, :], p=2, dim=1
            )

        batch_size = input_features.shape[0]
        questions = samples.get("question", [""] * batch_size)

        prompt_texts = []
        for i in range(batch_size):
            n_audio_tokens = audio_output_lengths[i].item()
            audio_placeholder = "<|AUDIO|>" * n_audio_tokens
            prompt = f"<|audio_bos|>{audio_placeholder}<|audio_eos|>"
            if questions[i]:
                prompt += f" {questions[i]}"
            prompt_texts.append(prompt)

        prompt_tokens = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
        ).to(device)

        target_tokens = self.tokenizer(
            texts_with_reg,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(device)

        target_ids = target_tokens.input_ids
        target_attn = target_tokens.attention_mask
        target_attn[:, 0] = 0  # hide <reg> column from other tokens

        embed_tokens = self._get_embed_tokens()
        prompt_embeds = embed_tokens(prompt_tokens.input_ids)
        target_embeds = embed_tokens(target_ids)

        audio_token_id = self.thinker.config.audio_token_id
        audio_mask = prompt_tokens.input_ids == audio_token_id
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(
            audio_mask_3d, audio_features_flat.to(prompt_embeds.dtype)
        )

        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        prompt_len = prompt_embeds.shape[1]
        total_len = prompt_len + target_embeds.shape[1]
        start_regress = prompt_len

        # 4D causal mask with <reg> column zeroed
        mask_4d = torch.ones(
            [batch_size, 1, total_len, total_len],
            dtype=inputs_embeds.dtype,
            device=device,
        )
        mask_4d = torch.tril(mask_4d, diagonal=0)

        mask_4d[:, 0, :, start_regress:] *= target_attn.unsqueeze(1).to(mask_4d.dtype)
        mask_4d[:, 0, start_regress, start_regress] = 1.0  # <reg> attends to itself

        prompt_attn = prompt_tokens.attention_mask
        mask_4d[:, 0, :, :prompt_len] *= prompt_attn.unsqueeze(1).to(mask_4d.dtype)

        mask_4d = 1.0 - mask_4d
        mask_4d = mask_4d.masked_fill(
            mask_4d > 0.5, float(torch.finfo(mask_4d.dtype).min)
        )

        # 3D position IDs for TMRoPE: (3, B, total_len)
        prompt_position_ids = torch.cumsum(prompt_attn, dim=1) - 1
        prompt_position_ids = prompt_position_ids.clamp(min=0)
        last_valid_prompt_pos = (prompt_attn.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
        target_position_ids = torch.cumsum(target_attn, dim=1) + last_valid_prompt_pos
        position_ids_1d = torch.cat(
            [prompt_position_ids, target_position_ids], dim=1
        )
        position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)

        targets = torch.full_like(target_ids, -100)
        for i in range(batch_size):
            valid_mask = target_attn[i].bool()
            valid_tokens = target_ids[i, valid_mask]
            targets[i, valid_mask] = torch.cat(
                [valid_tokens[1:], torch.tensor([-100], device=device)]
            )

        with self.maybe_autocast():
            text_outputs = self.thinker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden_state = text_outputs.last_hidden_state
            logits = self.thinker.lm_head(last_hidden_state)
            del text_outputs

            vocab_size = logits.shape[-1]
            logits_for_loss = logits[:, start_regress - 1:, :].reshape(-1, vocab_size)

            first_target = target_ids[:, 1:2].clone()
            first_target[first_target == self.reg_token_id] = -100
            full_targets = torch.cat([first_target, targets], dim=1).reshape(-1)
            loss_ntp = F.cross_entropy(
                logits_for_loss, full_targets, ignore_index=-100
            )

            reg_hidden = last_hidden_state[:, start_regress, :]
            reg_projected = self.conclusion_proj(reg_hidden)
            cos_sim = F.cosine_similarity(
                reg_projected, conclusion_embeds.to(reg_projected.dtype), dim=-1
            )
            loss_reg = (1 - cos_sim).mean()

            loss = loss_ntp + self.alpha * loss_reg

        out = {"loss": loss, "loss_ntp": loss_ntp, "loss_reg": loss_reg}

        if verbose:
            ntp_preds = logits_for_loss.argmax(dim=-1)
            ntp_mask = full_targets != -100
            ntp_correct = (ntp_preds[ntp_mask] == full_targets[ntp_mask]).float().sum()
            ntp_total = ntp_mask.sum().item()
            out.update({"ntp_correct": ntp_correct, "ntp_total": ntp_total})

        return out

    @classmethod
    def from_config(cls, config):
        model = cls(
            qwen25_omni_path=config.get(
                "qwen25_omni_path", "Qwen/Qwen2.5-Omni-7B"
            ),
            alpha=config.get("mutor_alpha", 1.0),
            max_txt_len=config.get("max_txt_len", 400),
            freeze_audio_tower=config.get("freeze_audio_tower", True),
            freeze_visual=config.get("freeze_visual", True),
            lora=config.get("lora", False),
            lora_rank=config.get("lora_rank", 8),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bert_conclusion_path=config.get("bert_conclusion_path") or None,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logger.info("Loading checkpoint from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        return model
