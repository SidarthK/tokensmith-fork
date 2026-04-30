#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in headless envs
    plt = None


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: Iterable[float]) -> float:
    cleaned = [value for value in values if value is not None]
    return mean(cleaned) if cleaned else 0.0


def metric_names(rows: List[Dict]) -> List[str]:
    names = set()
    for row in rows:
        for key in row.get("scores", {}):
            if key.endswith("_similarity") and key != "final_score":
                names.add(key[:-11])
    preferred_order = ["semantic", "keyword", "nli", "chunk_retrieval", "async_llm_judge"]
    ordered = [name for name in preferred_order if name in names]
    extras = sorted(name for name in names if name not in ordered)
    return ordered + extras


def summarize_rows(rows: List[Dict], metric_order: List[str]) -> Dict[str, float]:
    summary = {
        "questions": len(rows),
        "mean_final_score": safe_mean(row.get("scores", {}).get("final_score") for row in rows),
        "pass_rate": safe_mean(1 if row.get("passed") else 0 for row in rows),
        "mean_latency_seconds": safe_mean(row.get("latency_seconds") for row in rows),
        "mean_chunk_hit_ratio": safe_mean(row.get("scores", {}).get("chunk_retrieval_similarity") for row in rows),
    }
    for metric_name in metric_order:
        summary[f"mean_{metric_name}_similarity"] = safe_mean(
            row.get("scores", {}).get(f"{metric_name}_similarity") for row in rows
        )
    return summary


def grouped_summary(rows: List[Dict], field: str, metric_order: List[str]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "unknown")].append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append({field: key, **summarize_rows(group_rows, metric_order)})
    return summaries


def fmt_optional_float(value, digits=2):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def plot_bar_overall(run_root: Path, rows: List[Dict]) -> Path | None:
    if not plt or not rows:
        return None
    labels = [row["label"] for row in rows]
    values = [row["mean_final_score"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    ax.bar(labels, values, color="#2f6b8a")
    ax.set_ylabel("Mean Final Score")
    ax.set_title("Overall Final Score by Run")
    ax.set_ylim(0, max(1.0, max(values) + 0.1))
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    output = run_root / "overall_final_score.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_grouped_pass_rate(run_root: Path, family_rows: List[Dict]) -> Path | None:
    if not plt or not family_rows:
        return None
    families = sorted({row["question_family"] for row in family_rows})
    labels = sorted({row["label"] for row in family_rows})
    positions = list(range(len(families)))
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(families) * 1.1), 5.5))
    color_cycle = cycle(["#2f6b8a", "#7ea16b", "#c27b53", "#915e95", "#4d4d4d", "#c24f4f", "#c9a227", "#4aa3a2"])

    for idx, label in enumerate(labels):
        color = next(color_cycle)
        series = []
        for family in families:
            match = next((row for row in family_rows if row["label"] == label and row["question_family"] == family), None)
            series.append(match["pass_rate"] if match else 0.0)
        offsets = [pos + (idx - (len(labels) - 1) / 2) * width for pos in positions]
        ax.bar(offsets, series, width=width, label=label, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pass Rate by Question Family")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = run_root / "pass_rate_by_family.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_scatter(run_root: Path, question_rows: List[Dict], x_key: str, y_key: str, filename: str, title: str, x_label: str, y_label: str) -> Path | None:
    if not plt or not question_rows:
        return None
    labels = sorted({row["label"] for row in question_rows})
    color_cycle = cycle(["#2f6b8a", "#7ea16b", "#c27b53", "#915e95", "#4d4d4d", "#c24f4f", "#c9a227", "#4aa3a2"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label in labels:
        color = next(color_cycle)
        subset = [row for row in question_rows if row["label"] == label]
        xs = [row.get(x_key, 0.0) for row in subset]
        ys = [row.get(y_key, 0.0) for row in subset]
        ax.scatter(xs, ys, label=label, alpha=0.8, s=35, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = run_root / filename
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_heatmap(run_root: Path, question_rows: List[Dict]) -> Path | None:
    if not plt or not question_rows:
        return None
    labels = sorted({row["label"] for row in question_rows})
    questions = sorted({row["test_id"] for row in question_rows})
    matrix = []
    for question in questions:
        question_scores = []
        for label in labels:
            match = next((row for row in question_rows if row["test_id"] == question and row["label"] == label), None)
            question_scores.append(match["final_score"] if match else 0.0)
        matrix.append(question_scores)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), max(7, len(questions) * 0.45)))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(1.0, max(max(row) for row in matrix)))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions)
    ax.set_title("Per-Question Final Score Heatmap")
    fig.colorbar(image, ax=ax, label="Final Score")
    fig.tight_layout()
    output = run_root / "question_win_heatmap.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def build_question_rows(label: str, mode: str, metadata: Dict, rows: List[Dict], metric_order: List[str]) -> List[Dict]:
    question_rows = []
    for row in rows:
        flattened = {
            "label": label,
            "mode": mode,
            "lambda": metadata.get("coverage_mmr_lambda"),
            "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
            "test_id": row.get("test_id"),
            "question_family": row.get("question_family", row.get("question_category", "unknown")),
            "question_topic": row.get("question_topic", "general"),
            "passed": 1 if row.get("passed") else 0,
            "latency_seconds": row.get("latency_seconds", 0.0),
            "final_score": row.get("scores", {}).get("final_score", 0.0),
            "chunk_retrieval_similarity": row.get("scores", {}).get("chunk_retrieval_similarity", 0.0),
        }
        for metric_name in metric_order:
            flattened[f"{metric_name}_similarity"] = row.get("scores", {}).get(f"{metric_name}_similarity", 0.0)
        question_rows.append(flattened)
    return question_rows


def render_markdown(
    overall_rows: List[Dict],
    family_rows: List[Dict],
    topic_rows: List[Dict],
    metric_order: List[str],
    chart_paths: List[Path],
) -> str:
    lines = ["# Hard Multihop Sweep Summary", ""]
    if overall_rows:
        best_score = max(overall_rows, key=lambda row: row["mean_final_score"])
        best_pass = max(overall_rows, key=lambda row: row["pass_rate"])
        fastest = min(overall_rows, key=lambda row: row["mean_latency_seconds"])
        lines.extend([
            "## Narrative Summary",
            "",
            f"- Best mean final score: `{best_score['label']}` at `{best_score['mean_final_score']:.4f}`.",
            f"- Best pass rate: `{best_pass['label']}` at `{best_pass['pass_rate']:.4f}`.",
            f"- Lowest latency: `{fastest['label']}` at `{fastest['mean_latency_seconds']:.4f}` seconds.",
            "",
        ])

    lines.extend(["## Overall", ""])
    overall_headers = [
        "Label", "Mode", "Lambda", "Candidate Pool", "Questions", "Mean Final Score",
        "Pass Rate", "Mean Latency (s)", "Mean Chunk Hit Ratio",
    ] + [f"Mean {metric}" for metric in metric_order]
    lines.append("| " + " | ".join(overall_headers) + " |")
    lines.append("|" + "---|" * len(overall_headers))
    for row in overall_rows:
        values = [
            row["label"],
            row["mode"],
            fmt_optional_float(row.get("lambda")),
            "" if row.get("candidate_pool") is None else str(row.get("candidate_pool")),
            str(row["questions"]),
            f"{row['mean_final_score']:.4f}",
            f"{row['pass_rate']:.4f}",
            f"{row['mean_latency_seconds']:.4f}",
            f"{row['mean_chunk_hit_ratio']:.4f}",
        ] + [f"{row.get(f'mean_{metric}_similarity', 0.0):.4f}" for metric in metric_order]
        lines.append("| " + " | ".join(values) + " |")

    if family_rows:
        lines.extend(["", "## By Question Family", ""])
        family_headers = ["Label", "Family", "Questions", "Mean Final Score", "Pass Rate", "Mean Latency (s)", "Mean Chunk Hit Ratio"]
        lines.append("| " + " | ".join(family_headers) + " |")
        lines.append("|" + "---|" * len(family_headers))
        for row in family_rows:
            values = [
                row["label"],
                row["question_family"],
                str(row["questions"]),
                f"{row['mean_final_score']:.4f}",
                f"{row['pass_rate']:.4f}",
                f"{row['mean_latency_seconds']:.4f}",
                f"{row['mean_chunk_hit_ratio']:.4f}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    if topic_rows:
        lines.extend(["", "## By Topic", ""])
        topic_headers = ["Label", "Topic", "Questions", "Mean Final Score", "Pass Rate", "Mean Latency (s)"]
        lines.append("| " + " | ".join(topic_headers) + " |")
        lines.append("|" + "---|" * len(topic_headers))
        for row in topic_rows:
            values = [
                row["label"],
                row["question_topic"],
                str(row["questions"]),
                f"{row['mean_final_score']:.4f}",
                f"{row['pass_rate']:.4f}",
                f"{row['mean_latency_seconds']:.4f}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    if chart_paths:
        lines.extend(["", "## Generated Charts", ""])
        for chart_path in chart_paths:
            lines.append(f"- `{chart_path.name}`")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize hard multihop sweep outputs")
    parser.add_argument("--run-root", required=True, help="Root folder containing per-run subdirectories")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])

    all_rows: List[Dict] = []
    raw_runs: List[Tuple[Dict, List[Dict]]] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        results_path = run_dir / "benchmark_results.json"
        if not metadata_path.exists() or not results_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = load_jsonl(results_path)
        raw_runs.append((metadata, rows))
        all_rows.extend(rows)

    metrics = metric_names(all_rows)
    overall_rows: List[Dict] = []
    family_rows: List[Dict] = []
    topic_rows: List[Dict] = []
    question_rows: List[Dict] = []

    for metadata, rows in raw_runs:
        label = metadata.get("label")
        mode = metadata.get("mode", "")
        overall_rows.append({
            "label": label,
            "mode": mode,
            "lambda": metadata.get("coverage_mmr_lambda"),
            "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
            **summarize_rows(rows, metrics),
        })
        for family_summary in grouped_summary(rows, "question_family", metrics):
            family_rows.append({
                "label": label,
                "mode": mode,
                "lambda": metadata.get("coverage_mmr_lambda"),
                "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
                **family_summary,
            })
        for topic_summary in grouped_summary(rows, "question_topic", metrics):
            topic_rows.append({
                "label": label,
                "mode": mode,
                "lambda": metadata.get("coverage_mmr_lambda"),
                "candidate_pool": metadata.get("coverage_mmr_candidate_pool"),
                **topic_summary,
            })
        question_rows.extend(build_question_rows(label, mode, metadata, rows, metrics))

    overall_rows.sort(key=lambda row: row["label"])
    family_rows.sort(key=lambda row: (row["question_family"], row["label"]))
    topic_rows.sort(key=lambda row: (row["question_topic"], row["label"]))
    question_rows.sort(key=lambda row: (row["test_id"], row["label"]))

    overall_fieldnames = [
        "label", "mode", "lambda", "candidate_pool", "questions", "mean_final_score",
        "pass_rate", "mean_latency_seconds", "mean_chunk_hit_ratio",
    ] + [f"mean_{metric}_similarity" for metric in metrics]
    family_fieldnames = [
        "label", "mode", "lambda", "candidate_pool", "question_family", "questions",
        "mean_final_score", "pass_rate", "mean_latency_seconds", "mean_chunk_hit_ratio",
    ] + [f"mean_{metric}_similarity" for metric in metrics]
    topic_fieldnames = [
        "label", "mode", "lambda", "candidate_pool", "question_topic", "questions",
        "mean_final_score", "pass_rate", "mean_latency_seconds", "mean_chunk_hit_ratio",
    ] + [f"mean_{metric}_similarity" for metric in metrics]
    question_fieldnames = [
        "label", "mode", "lambda", "candidate_pool", "test_id", "question_family", "question_topic",
        "passed", "latency_seconds", "final_score", "chunk_retrieval_similarity",
    ] + [f"{metric}_similarity" for metric in metrics]

    overall_csv = run_root / "sweep_summary.csv"
    family_csv = run_root / "family_breakdown.csv"
    topic_csv = run_root / "topic_breakdown.csv"
    question_csv = run_root / "question_level_scores.csv"

    write_csv(overall_csv, overall_rows, overall_fieldnames)
    write_csv(family_csv, family_rows, family_fieldnames)
    write_csv(topic_csv, topic_rows, topic_fieldnames)
    write_csv(question_csv, question_rows, question_fieldnames)

    chart_paths = []
    for plotter in (
        lambda: plot_bar_overall(run_root, overall_rows),
        lambda: plot_grouped_pass_rate(run_root, family_rows),
        lambda: plot_scatter(run_root, question_rows, "latency_seconds", "final_score", "latency_vs_final_score.png", "Latency vs Final Score", "Latency (s)", "Final Score"),
        lambda: plot_scatter(run_root, question_rows, "chunk_retrieval_similarity", "final_score", "chunk_hit_vs_final_score.png", "Chunk Hit Ratio vs Final Score", "Chunk Retrieval Similarity", "Final Score"),
        lambda: plot_heatmap(run_root, question_rows),
    ):
        chart_path = plotter()
        if chart_path is not None:
            chart_paths.append(chart_path)

    output_md = run_root / "sweep_summary.md"
    output_md.write_text(render_markdown(overall_rows, family_rows, topic_rows, metrics, chart_paths), encoding="utf-8")

    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote csv summary: {overall_csv}")
    print(f"Wrote family breakdown: {family_csv}")
    print(f"Wrote topic breakdown: {topic_csv}")
    print(f"Wrote question-level scores: {question_csv}")
    if chart_paths:
        for chart_path in chart_paths:
            print(f"Wrote chart: {chart_path}")
    elif plt is None:
        print("Skipping chart generation because matplotlib is unavailable.")


if __name__ == "__main__":
    main()
