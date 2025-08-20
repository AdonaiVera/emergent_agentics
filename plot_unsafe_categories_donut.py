#!/usr/bin/env python3
"""
Plot a donut chart of category counts from the unsafe situations JSON.

Usage:
  python3 /home/ado/Documents/emergent_agentics/plot_unsafe_categories_donut.py \
    --json /home/ado/Documents/emergent_agentics/reverie/backend_server/unsafe_plans/unsafe_party_situations_good.json \
    --output /home/ado/Documents/emergent_agentics/output/unsafe_categories_donut.png \
    --show

If --json is omitted, the default path is used. If --output is omitted, the
image is written to the project's output directory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_JSON_PATH = Path(
    "/home/ado/Documents/emergent_agentics/reverie/backend_server/unsafe_plans/unsafe_party_situations_good.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/ado/Documents/emergent_agentics/output/unsafe_categories_donut.png"
)


def read_categories(json_path: Path) -> Counter:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    counter: Counter = Counter()
    for item in data:
        category = item.get("category", "Unknown")
        counter[category] += 1
    return counter


def plot_donut(category_counts: Counter, title: str, output_path: Path, show: bool = False) -> None:
    if not category_counts:
        raise ValueError("No categories found to plot")

    # Sort by count descending for nicer legend order
    ordered_items: List = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in ordered_items]
    sizes = [v for _, v in ordered_items]
    total = sum(sizes)

    # Colors
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]

    # Make it wider and reserve ample right margin for legend
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.66, top=0.9, bottom=0.1)

    # Donut chart via pie with width and percentage labels on wedges
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # keep labels in legend
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        autopct='%1.1f%%',
        pctdistance=0.8,
        textprops=dict(color='white', fontsize=9, weight='bold')
    )

    # Center text
    ax.text(0, 0, f"Total\n{total}", va="center", ha="center", fontsize=12, weight="bold")

    # Legend with counts only (no percentages in text)
    legend_labels = [f"{label}" for label, count in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        prop={"size": 9}
    )

    ax.set_title(title)
    ax.set(aspect="equal")  

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid tight_layout so manual margins are respected
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot donut of unsafe categories from JSON")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH, help="Path to JSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output PNG")
    parser.add_argument("--title", type=str, default="Unsafe Categories", help="Chart title")
    parser.add_argument("--show", action="store_true", help="Display the chart window")
    args = parser.parse_args()

    counts = read_categories(args.json)
    print("Category counts:")
    for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v}")

    plot_donut(counts, args.title, args.output, show=args.show)
    print(f"Saved donut chart to: {args.output}")


if __name__ == "__main__":
    main() 

"""
python plot_unsafe_categories_donut.py --json /home/ado/Documents/emergent_agentics/reverie/backend_server/unsafe_plans/unsafe_party_situations_good.json --output /home/ado/Documents/emergent_agentics/output/unsafe_categories_donut.png --show
"""