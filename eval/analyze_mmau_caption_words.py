#!/usr/bin/env python3
"""
MMAU: word counts inside <CAPTION>...</CAPTION> per model, split by correct vs incorrect
(string_match vs gold answer). Writes JSON/CSV summaries and matplotlib figures.

Correlation note: longer captions on correct items need not imply causation — models may
write more when confident, or verbose chains may co-occur with better listening.

Run from the SALMONN repo root (so imports are not required; string_match is inlined):

  python eval/analyze_mmau_caption_words.py \\
    --out-dir ./mmau_caption_analysis_out \\
    --mutor /path/to/mutor_answers.json \\
    --salmonn-pretrained /path/to/salmonn_pretrained_answers.json \\
    --salmonn-sft /path/to/salmonn_sft_seed1_answers.json \\
    --spare /path/to/spare_seed1_answers.json

Example (local CoT_analysis folder on this machine):

  python eval/analyze_mmau_caption_words.py \\
    --out-dir ./mmau_caption_analysis_out \\
    --mutor ~/Desktop/SPARE/CoT_analysis/mutor_answers.json \\
    --salmonn-pretrained ~/Desktop/SPARE/CoT_analysis/salmonn_pretrained_answers.json \\
    --salmonn-sft ~/Desktop/SPARE/CoT_analysis/salmonn_sft_seed1_answers.json \\
    --spare ~/Desktop/SPARE/CoT_analysis/spare_seed1_answers.json \\
    --models spare,salmonn_sft,mutor --ecdf
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# --- Inlined from evaluate_qwen_mmau.py (keep in sync) ---


def string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def extract_caption(hyp: str | None) -> str | None:
    if not hyp or not isinstance(hyp, str):
        return None
    m = re.search(r"<CAPTION>\s*(.*?)\s*</CAPTION>", hyp, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower()))


def load_records(path: str) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        if "id" not in row:
            continue
        out[str(row["id"])] = row
    return out


def _stats(xs: list[int]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
    }


def _default_json_dir() -> str | None:
    base = "/Users/francescobonzi/Desktop/SPARE/CoT_analysis"
    if os.path.isdir(base) and os.path.isfile(os.path.join(base, "spare_seed1_answers.json")):
        return base
    return None


def main() -> None:
    ddir = _default_json_dir()
    parser = argparse.ArgumentParser(description="Caption word counts vs correctness on MMAU JSON outputs.")
    parser.add_argument("--mutor", type=str, default=os.path.join(ddir, "mutor_answers.json") if ddir else None)
    parser.add_argument("--salmonn-pretrained", type=str, default=os.path.join(ddir, "salmonn_pretrained_answers.json") if ddir else None)
    parser.add_argument("--salmonn-sft", type=str, default=os.path.join(ddir, "salmonn_sft_seed1_answers.json") if ddir else None)
    parser.add_argument("--spare", type=str, default=os.path.join(ddir, "spare_seed1_answers.json") if ddir else None)
    parser.add_argument("--out-dir", type=str, default="mmau_caption_analysis_out")
    parser.add_argument(
        "--models",
        type=str,
        default="mutor,salmonn_pretrained,salmonn_sft,spare",
        help="Comma-separated keys: mutor,salmonn_pretrained,salmonn_sft,spare",
    )
    parser.add_argument(
        "--exclude-missing-caption",
        action="store_true",
        help="Only use sample ids where every model in --models has a non-empty <CAPTION> (fair cross-model comparison).",
    )
    parser.add_argument("--ecdf", action="store_true", help="Also write caption_words_ecdf.png (SPARE + salmonn_sft).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    paths = {
        "mutor": args.mutor,
        "salmonn_pretrained": args.salmonn_pretrained,
        "salmonn_sft": args.salmonn_sft,
        "spare": args.spare,
    }
    for k, p in paths.items():
        if not p or not os.path.isfile(p):
            raise SystemExit(f"Missing or invalid path for {k}: {p!r}. Pass all four --*-json paths.")

    model_order = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"mutor", "salmonn_pretrained", "salmonn_sft", "spare"}
    for m in model_order:
        if m not in valid:
            raise SystemExit(f"Unknown model key {m!r}. Use: {sorted(valid)}")

    display_name = {
        "mutor": "MuTor",
        "salmonn_pretrained": "SALMONN (pre)",
        "salmonn_sft": "SALMONN (SFT)",
        "spare": "SPARE",
    }

    loaded = {m: load_records(paths[m]) for m in valid}
    id_sets = [set(loaded[m]) for m in valid]
    inter = set.intersection(*id_sets)
    if not inter:
        raise SystemExit("No common ids across all four files.")

    ids_sorted = sorted(inter)
    rows_long: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, list[int]]] = {m: {"correct": [], "incorrect": []} for m in model_order}
    missing: dict[str, int] = {m: 0 for m in model_order}

    for sid in ids_sorted:
        recs = {m: loaded[m][sid] for m in valid}
        gold = recs["spare"]
        answer = str(gold.get("answer", ""))
        choices = gold.get("choices") or []
        if not isinstance(choices, list):
            choices = []

        caps: dict[str, str | None] = {}
        for m in model_order:
            r = recs[m]
            hyp = r.get("hyp")
            caps[m] = extract_caption(hyp if isinstance(hyp, str) else None)

        all_have_caption = all(caps[m] is not None and caps[m] != "" for m in model_order)
        skip_for_buckets = args.exclude_missing_caption and not all_have_caption

        for m in model_order:
            r = recs[m]
            cap = caps[m]
            pred = r.get("model_output")
            pred_s = pred if isinstance(pred, str) else ("" if pred is None else str(pred))
            ok = string_match(answer, pred_s, [str(x) for x in choices])
            if cap is None:
                missing[m] += 1
                words: int | None = None
            else:
                words = count_words(cap)

            rows_long.append(
                {
                    "id": sid,
                    "model": m,
                    "correct": int(ok),
                    "caption_words": words,
                    "missing_caption": int(cap is None),
                }
            )

            if skip_for_buckets:
                continue
            if cap is None:
                continue
            assert words is not None
            if ok:
                buckets[m]["correct"].append(words)
            else:
                buckets[m]["incorrect"].append(words)

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary: dict[str, Any] = {
        "n_ids": len(ids_sorted),
        "paths": paths,
        "models_analyzed": model_order,
        "exclude_missing_caption": bool(args.exclude_missing_caption),
        "missing_caption_counts": missing,
        "per_model": {},
        "note_pretrained": "salmonn_pretrained outputs often lack <CAPTION> tags; caption stats may be empty for that model.",
    }

    for m in model_order:
        c_arr = buckets[m]["correct"]
        i_arr = buckets[m]["incorrect"]
        summary["per_model"][m] = {
            "missing_caption_total": missing[m],
            "correct_caption_stats": _stats(c_arr),
            "incorrect_caption_stats": _stats(i_arr),
        }

    csv_path = os.path.join(args.out_dir, "caption_word_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "correct", "caption_words", "missing_caption"])
        w.writeheader()
        for row in rows_long:
            if row["model"] not in model_order:
                continue
            w.writerow(row)

    plot_buckets = buckets

    plot_models = [m for m in model_order if plot_buckets[m]["correct"] or plot_buckets[m]["incorrect"]]
    summary["models_plotted"] = plot_models
    if not plot_models:
        raise SystemExit("No <CAPTION> text found for any model in --models (cannot plot).")

    json_path = os.path.join(args.out_dir, "caption_word_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure A: grouped boxplot (skip empty buckets — matplotlib rejects empty lists)
    n_m = len(plot_models)
    x = np.arange(n_m, dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_m), 5))
    pos_c = x - width / 2
    pos_i = x + width / 2
    first_green = first_red = None
    for mi, m in enumerate(plot_models):
        xc = plot_buckets[m]["correct"]
        xi = plot_buckets[m]["incorrect"]
        if xc:
            bp = ax.boxplot(
                [xc],
                positions=[pos_c[mi]],
                widths=width,
                patch_artist=True,
                showfliers=True,
            )
            for b in bp["boxes"]:
                b.set_facecolor("#4CAF50")
                b.set_alpha(0.55)
            if first_green is None:
                first_green = bp["boxes"][0]
        if xi:
            bp = ax.boxplot(
                [xi],
                positions=[pos_i[mi]],
                widths=width,
                patch_artist=True,
                showfliers=True,
            )
            for b in bp["boxes"]:
                b.set_facecolor("#F44336")
                b.set_alpha(0.55)
            if first_red is None:
                first_red = bp["boxes"][0]
    legend_handles = []
    legend_labels = []
    if first_green is not None:
        legend_handles.append(first_green)
        legend_labels.append("Correct")
    if first_red is not None:
        legend_handles.append(first_red)
        legend_labels.append("Incorrect")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name[m] for m in plot_models])
    ax.set_ylabel("Caption word count")
    note = "per-model caption present" if not args.exclude_missing_caption else "intersection: all models have caption"
    ax.set_title(f"MMAU caption length vs correctness ({note}; n_ids={len(ids_sorted)})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "caption_words_boxplot.png"), dpi=args.dpi)
    plt.close(fig)

    # Figure B: mean bars
    fig2, ax2 = plt.subplots(figsize=(max(6, 1.2 * n_m), 5))
    means_c = [statistics.mean(plot_buckets[m]["correct"]) if plot_buckets[m]["correct"] else 0.0 for m in plot_models]
    means_i = [statistics.mean(plot_buckets[m]["incorrect"]) if plot_buckets[m]["incorrect"] else 0.0 for m in plot_models]
    ax2.bar(pos_c, means_c, width=width, label="Correct (mean)", color="#4CAF50", alpha=0.8)
    ax2.bar(pos_i, means_i, width=width, label="Incorrect (mean)", color="#F44336", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([display_name[m] for m in plot_models])
    ax2.set_ylabel("Mean caption word count")
    ax2.set_title(f"MMAU mean caption length ({note})")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "caption_words_mean_bars.png"), dpi=args.dpi)
    plt.close(fig2)

    # Figure C: ECDF SPARE + SFT
    if args.ecdf:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ecdf_models = [m for m in ("spare", "salmonn_sft") if m in plot_models]
        if not ecdf_models:
            ecdf_models = plot_models[:2]
        colours = plt.cm.tab10(np.linspace(0, 0.9, max(4, 2 * len(ecdf_models))))
        ci = 0
        for m in ecdf_models:
            for label, arr, ls in [
                (f"{display_name[m]} correct", plot_buckets[m]["correct"], "-"),
                (f"{display_name[m]} incorrect", plot_buckets[m]["incorrect"], "--"),
            ]:
                if not arr:
                    continue
                xs = np.sort(np.array(arr, dtype=float))
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax3.step(
                    np.concatenate([[xs[0]], xs]),
                    np.concatenate([[0.0], ys]),
                    where="post",
                    label=label,
                    linestyle=ls,
                    color=colours[ci],
                )
                ci += 1
        ax3.set_xlabel("Caption word count")
        ax3.set_ylabel("ECDF")
        ax3.set_title("Caption length ECDF (caption-present samples)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(os.path.join(plots_dir, "caption_words_ecdf.png"), dpi=args.dpi)
        plt.close(fig3)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote plots under {plots_dir}")
    for m in model_order:
        pc = plot_buckets[m]["correct"]
        pi = plot_buckets[m]["incorrect"]
        mc = statistics.mean(pc) if pc else None
        mi = statistics.mean(pi) if pi else None
        print(
            f"  {m}: caption-bucket correct n={len(pc)} mean={mc if mc is not None else 'n/a'} | "
            f"incorrect n={len(pi)} mean={mi if mi is not None else 'n/a'} | missing_caption_rows={missing[m]}"
        )


if __name__ == "__main__":
    main()
