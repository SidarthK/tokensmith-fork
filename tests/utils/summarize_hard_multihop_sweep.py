#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else 0.0


def summarize_rows(rows):
    final_scores = [r.get("scores", {}).get("final_score") for r in rows]
    pass_flags = [1 if r.get("passed") else 0 for r in rows]
    latencies = [r.get("latency_seconds") for r in rows]
    chunk_hits = [r.get("scores", {}).get("chunk_retrieval_similarity") for r in rows]
    return {
        "questions": len(rows),
        "mean_final_score": safe_mean(final_scores),
        "pass_rate": safe_mean(pass_flags),
        "mean_latency_seconds": safe_mean(latencies),
        "mean_chunk_hit_ratio": safe_mean(chunk_hits),
    }


def fmt_optional_float(value, digits=2):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summarize_by_category(rows):
    cats = {}
    for row in rows:
        cats.setdefault(row.get("question_category", "other"), []).append(row)
    return {cat: summarize_rows(group) for cat, group in cats.items()}


def main():
    parser = argparse.ArgumentParser(description="Summarize hard multihop sweep outputs")
    parser.add_argument("--run-root", required=True, help="Root folder containing per-run subdirectories")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])

    rows = []
    category_rows = []

    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        results_path = run_dir / "benchmark_results.json"
        if not metadata_path.exists() or not results_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_rows = load_jsonl(results_path)
        summary = summarize_rows(run_rows)
        rows.append({
            "label": metadata.get("label", run_dir.name),
            "mode": metadata.get("mode", ""),
            "lambda": metadata.get("coverage_mmr_lambda"),
            "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
            **summary,
        })

        for cat, cat_summary in summarize_by_category(run_rows).items():
            category_rows.append({
                "label": metadata.get("label", run_dir.name),
                "mode": metadata.get("mode", ""),
                "category": cat,
                "lambda": metadata.get("coverage_mmr_lambda"),
                "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
                **cat_summary,
            })

    output_csv = run_root / "sweep_summary.csv"
    output_md = run_root / "sweep_summary.md"

    fieldnames = [
        "label", "mode", "lambda", "candidate_pool", "questions",
        "mean_final_score", "pass_rate", "mean_latency_seconds", "mean_chunk_hit_ratio",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda row: (row["mode"] == "coverage_mmr", row["label"]))
    category_rows.sort(key=lambda row: (row["label"], row["category"]))

    lines = ["# Hard Multihop Sweep Summary", "", "## Overall", ""]
    lines.append("| Label | Mode | Lambda | Candidate Pool | Questions | Mean Final Score | Pass Rate | Mean Latency (s) | Mean Chunk Hit Ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['mode']} | {fmt_optional_float(row['lambda'])} | "
            f"{'' if row['candidate_pool'] is None else row['candidate_pool']} | {row['questions']} | "
            f"{row['mean_final_score']:.4f} | {row['pass_rate']:.4f} | {row['mean_latency_seconds']:.4f} | {row['mean_chunk_hit_ratio']:.4f} |"
        )

    if category_rows:
        lines.extend(["", "## By Category", ""])
        lines.append("| Label | Category | Mean Final Score | Pass Rate | Mean Latency (s) | Mean Chunk Hit Ratio |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in category_rows:
            lines.append(
                f"| {row['label']} | {row['category']} | {row['mean_final_score']:.4f} | "
                f"{row['pass_rate']:.4f} | {row['mean_latency_seconds']:.4f} | {row['mean_chunk_hit_ratio']:.4f} |"
            )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote csv summary: {output_csv}")


if __name__ == "__main__":
    main()
