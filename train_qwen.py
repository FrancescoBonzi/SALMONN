"""
Standalone training script for MuToR BERT Conclusion on Qwen2.5-Omni.

Usage:
    python train_qwen2audio.py --cfg-path recipes/qwen2audio/mutor_bert_conclusion.yaml

    # With overrides:
    python train_qwen2audio.py --cfg-path recipes/qwen2audio/mutor_bert_conclusion.yaml \\
        --options model.mutor_alpha=0.5 run.batch_size_train=4
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time
import datetime
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from config import Config
from dist_utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    main_process,
)
from logger import MetricLogger, SmoothedValue
from utils import get_dataloader, prepare_sample, now, setup_logger
from optims import get_optimizer, LinearWarmupCosineLRScheduler
from dataset import Qwen2AudioDataset
from models.qwen import MutorBERTConclusionQwen25Omni


def parse_args():
    parser = argparse.ArgumentParser(description="Train MuToR BERT Conclusion on Qwen2.5-Omni")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override settings in the config, key-value pair in xxx=yyy format",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Qwen25OmniTrainer:
    def __init__(self, cfg):
        self.config = cfg
        run_config = cfg.config.run
        model_config = cfg.config.model
        data_config = cfg.config.datasets

        # Output / logging
        job_id = now()
        self.output_dir = Path(run_config.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(self.output_dir)

        # Device
        self.device = torch.device(run_config.device)
        self.use_distributed = run_config.use_distributed
        self.start_epoch = 0
        self.max_epoch = run_config.optims.max_epoch

        # Build model
        logging.info("Building MutorBERTConclusionQwen25Omni...")
        self._model = MutorBERTConclusionQwen25Omni.from_config(model_config)
        self._model.to(self.device)

        if self.use_distributed:
            self.model = DDP(self._model, device_ids=[run_config.gpu], find_unused_parameters=True)
        else:
            self.model = self._model

        # Build datasets
        whisper_path = data_config.get("whisper_path", model_config.get("qwen25_omni_path", "Qwen/Qwen2.5-Omni-7B"))
        datasets = {
            "train": Qwen2AudioDataset(data_config.train_ann_path, whisper_path),
            "valid": Qwen2AudioDataset(data_config.valid_ann_path, whisper_path),
            "test": Qwen2AudioDataset(data_config.test_ann_path, whisper_path),
        }

        # Dataloaders
        self.train_loader = get_dataloader(datasets["train"], run_config, is_train=True, use_distributed=self.use_distributed)
        self.valid_loader = get_dataloader(datasets["valid"], run_config, is_train=False, use_distributed=self.use_distributed)
        self.test_loader = get_dataloader(datasets["test"], run_config, is_train=False, use_distributed=self.use_distributed)

        # AMP scaler
        self.use_amp = run_config.get("amp", False)
        model_uses_bf16 = any(p.dtype == torch.bfloat16 for p in self.model.parameters())
        if self.use_amp and not model_uses_bf16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        if model_uses_bf16:
            logging.info("Using bfloat16 - GradScaler disabled")

        # Optimizer & scheduler
        self.iters_per_epoch = (
            len(self.train_loader)
            if run_config.get("epoch_based", False)
            else run_config.iters_per_epoch
        )
        self.optimizer = get_optimizer(self.model, run_config.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=run_config.optims.min_lr,
            init_lr=run_config.optims.init_lr,
            warmup_steps=run_config.optims.warmup_steps,
            warmup_start_lr=run_config.optims.get("warmup_start_lr", -1),
        )

        self.evaluate_only = run_config.get("evaluate", False)
        self.save_only_best = run_config.get("save_only_best", False)
        self.log_freq = run_config.get("log_freq", 10)
        self.accum_grad_iters = run_config.get("accum_grad_iters", 1)

        self.log_config()

    def unwrap_model(self):
        if self.use_distributed:
            return self.model.module
        return self.model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_ntp", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_reg", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info("Start training epoch %d, %d iters per epoch.", epoch, self.iters_per_epoch)
        header = f"Train: data epoch: [{epoch}]"

        for i in metric_logger.log_every(
            range(self.iters_per_epoch),
            self.log_freq,
            header=header,
            logger=self.log_writer,
            start_step=epoch * self.iters_per_epoch,
        ):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=(self.device.type == "cuda"))

            self.scheduler.step(cur_epoch=epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                outputs = self.model(samples)
                loss = outputs["loss"]
                loss_ntp = outputs.get("loss_ntp", loss)
                loss_reg = outputs.get("loss_reg", torch.tensor(0.0))

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.accum_grad_iters == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_ntp=loss_ntp.item() if hasattr(loss_ntp, "item") else loss_ntp)
            metric_logger.update(loss_reg=loss_reg.item() if hasattr(loss_reg, "item") else loss_reg)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: %s", metric_logger.global_avg())
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split="valid", decode=False, save_json=False):
        model = self.unwrap_model()
        model.eval()

        dataloader = getattr(self, f"{split}_loader", None)
        assert dataloader is not None, f"{split}_loader does not exist."

        metric_logger = MetricLogger(delimiter="  ")
        header = f"Eval: data epoch: [{epoch}]"

        results = []
        for samples in metric_logger.log_every(dataloader, self.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=(self.device.type == "cuda"))

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                forward_result = model(samples, verbose=True)

            loss = forward_result.get("loss", 0)
            loss_ntp = forward_result.get("loss_ntp", loss)
            loss_reg = forward_result.get("loss_reg", 0)
            ntp_correct = forward_result.get("ntp_correct", 0)
            ntp_total = forward_result.get("ntp_total", 1)

            res = {
                "id": samples["id"],
                "ground_truth": samples.get("text", samples.get("answer", "")),
                "loss": loss.item() if hasattr(loss, "item") else loss,
                "loss_ntp": loss_ntp.item() if hasattr(loss_ntp, "item") else loss_ntp,
                "loss_reg": loss_reg.item() if hasattr(loss_reg, "item") else loss_reg,
                "acc": (ntp_correct / ntp_total).item() if hasattr(ntp_correct, "item") and ntp_total > 0 else 0,
                "total": ntp_total,
            }

            if decode:
                text = model.generate(samples, self.config.config.generate)
                res["text"] = text

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self._save_result(results, self.output_dir, f"eval_{split}_epoch_{epoch}")

        # Aggregate
        agg = {
            "loss": torch.tensor(0.0).cuda(),
            "n_sample": torch.tensor(0.0).cuda(),
            "correct": torch.tensor(0.0).cuda(),
            "n_token": torch.tensor(0.0).cuda(),
        }
        for item in results:
            n = len(item["id"])
            agg["loss"] += item["loss"] * n
            agg["n_sample"] += n
            agg["correct"] += item["acc"] * item["total"]
            agg["n_token"] += item["total"]

        if is_dist_avail_and_initialized():
            for v in agg.values():
                dist.all_reduce(v)

        ret = {
            "loss": (agg["loss"] / agg["n_sample"]).item(),
            "agg_metrics": (agg["correct"] / agg["n_token"]).item() if agg["n_token"] > 0 else 0,
        }
        return ret

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self._log_stats(train_stats, split_name="train")

            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
            if valid_log is not None and is_main_process():
                agg_metrics = valid_log["agg_metrics"]
                if agg_metrics > best_agg_metric:
                    best_agg_metric = agg_metrics
                    best_epoch = cur_epoch
                    self._save_checkpoint(cur_epoch, is_best=True)

                valid_log["best_epoch"] = best_epoch
                self._log_stats(valid_log, split_name="valid")

            if not self.save_only_best:
                self._save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        if self.evaluate_only:
            self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        logging.info("Training time %s", str(datetime.timedelta(seconds=int(total_time))))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def _log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {f"{split_name}_{k}": v for k, v in stats.items()}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        model_no_ddp = self.unwrap_model()
        param_grad_dic = {
            k: v.requires_grad for k, v in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic and not param_grad_dic[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        tag = "best" if is_best else cur_epoch
        save_to = os.path.join(self.output_dir, f"checkpoint_{tag}.pth")
        logging.info("Saving checkpoint at epoch %d to %s.", cur_epoch, save_to)
        torch.save(save_obj, save_to)

    def _save_result(self, result, result_dir, filename):
        result_file = os.path.join(result_dir, f"{filename}_rank{get_rank()}.json")
        final_result_file = os.path.join(result_dir, f"{filename}.json")

        json.dump(result, open(result_file, "w"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            merged = []
            for rank in range(get_world_size()):
                rf = os.path.join(result_dir, f"{filename}_rank{rank}.json")
                merged += json.load(open(rf, "r"))
            json.dump(merged, open(final_result_file, "w"), ensure_ascii=False)
            logging.info("Result file saved to %s", final_result_file)


def main():
    job_id = now()

    cfg = Config(parse_args())
    run_config = cfg.config.run

    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger()

    cfg.pretty_print()

    trainer = Qwen25OmniTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
