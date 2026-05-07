from __future__ import annotations

import os
from pathlib import Path
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/var/folders/cq/7gg8kh8d48d5cvmqdj5lcw440000gn/T/matplotlib-codex-project1",
)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path("/Users/joeysandoval/Desktop/dsc140b/finalproject")
DATA_PATH = Path("/Users/joeysandoval/Desktop/dsc106/dsc106project1/grocerydb.csv")
OUT_DIR = ROOT / "project1_outputs"


STORE_COLORS = {
    "Target": "#d62828",
    "Walmart": "#1d4e89",
    "WholeFoods": "#2d6a4f",
}

CLASS_COLORS = {
    0.0: "#95d5b2",
    1.0: "#74c69d",
    2.0: "#f4d35e",
    3.0: "#ee6c4d",
}

CLASS_LABELS = {
    0.0: "Unprocessed / minimally processed",
    1.0: "Processed culinary ingredients",
    2.0: "Processed foods",
    3.0: "Ultra-processed foods",
}

CATEGORY_LABELS = {
    "baking": "Baking",
    "sausage-bacon": "Sausage / bacon",
    "drink-shakes-other": "Shakes / meal drinks",
    "pastry-chocolate-candy": "Candy / pastry",
    "soup-stew": "Soup / stew",
    "produce-packaged": "Packaged produce",
    "cereal": "Cereal",
    "dairy-yogurt-drink": "Drinkable yogurt",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["FPro_class"] = df["FPro_class"].astype(float)
    return df


def build_overall_mix(df: pd.DataFrame) -> pd.DataFrame:
    mix = pd.crosstab(df["store"], df["FPro_class"], normalize="index")
    mix = mix.reindex(["Target", "Walmart", "WholeFoods"])
    return mix


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["category", "store"]).size().unstack(fill_value=0)
    common = counts[(counts > 40).all(axis=1)].index

    rows = []
    for category in common:
        group = (
            df[df["category"] == category]
            .groupby("store")
            .agg(
                items=("name", "size"),
                ultra_share=("FPro_class", lambda s: (s == 3.0).mean()),
            )
            .reset_index()
        )
        values = {row["store"]: row["ultra_share"] for _, row in group.iterrows()}
        advantage = ((values["Target"] + values["Walmart"]) / 2) - values["WholeFoods"]
        rows.append(
            {
                "category": category,
                "Target": values["Target"],
                "Walmart": values["Walmart"],
                "WholeFoods": values["WholeFoods"],
                "wf_advantage": advantage,
            }
        )

    summary = pd.DataFrame(rows).sort_values("wf_advantage", ascending=False)
    return summary[summary["category"].isin(CATEGORY_LABELS)].copy()


def draw_stack(ax: plt.Axes, mix: pd.DataFrame) -> None:
    stores = ["Target", "Walmart", "WholeFoods"]
    bottoms = [0.0] * len(stores)

    for cls in [0.0, 1.0, 2.0, 3.0]:
        values = mix[cls].to_list()
        ax.bar(
            stores,
            values,
            bottom=bottoms,
            color=CLASS_COLORS[cls],
            edgecolor="white",
            linewidth=1.5,
            width=0.62,
            label=CLASS_LABELS[cls],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of each store's catalog", fontsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    for idx, store in enumerate(stores):
        ultra_share = mix.loc[store, 3.0]
        ax.text(
            idx,
            0.96,
            f"{ultra_share:.0%}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        ax.text(
            idx,
            0.92,
            "ultra-processed",
            ha="center",
            va="center",
            fontsize=8.8,
            color="white",
        )


def draw_dumbbell(ax: plt.Axes, category_summary: pd.DataFrame) -> None:
    data = category_summary.sort_values("wf_advantage", ascending=True)
    y_positions = range(len(data))

    for y, (_, row) in zip(y_positions, data.iterrows()):
        values = [row["Target"] * 100, row["Walmart"] * 100, row["WholeFoods"] * 100]
        ax.plot(
            [min(values), max(values)],
            [y, y],
            color="#c7c7c7",
            linewidth=2.5,
            zorder=1,
        )
        for store in ["Target", "Walmart", "WholeFoods"]:
            ax.scatter(
                row[store] * 100,
                y,
                s=110,
                color=STORE_COLORS[store],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        ax.text(
            row["WholeFoods"] * 100 - 2.2,
            y + 0.19,
            f"{row['WholeFoods']:.0%}",
            ha="right",
            va="center",
            fontsize=9,
            color=STORE_COLORS["WholeFoods"],
        )
        ax.text(
            max(row["Target"], row["Walmart"]) * 100 + 2.2,
            y + 0.19,
            f"{max(row['Target'], row['Walmart']):.0%}",
            ha="left",
            va="center",
            fontsize=9,
            color="#555555",
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([CATEGORY_LABELS[cat] for cat in data["category"]], fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Ultra-processed share within category", fontsize=11)


def save_visualization(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    mix = build_overall_mix(df)
    category_summary = build_category_summary(df)

    fig = plt.figure(figsize=(15, 9.5), facecolor="#faf7f2")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.18, 0.82],
        width_ratios=[1, 1.15],
        hspace=0.08,
        wspace=0.26,
    )

    title_ax = fig.add_subplot(gs[0, :])
    left_ax = fig.add_subplot(gs[1, 0], facecolor="#faf7f2")
    right_ax = fig.add_subplot(gs[1, 1], facecolor="#faf7f2")

    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.97,
        "Whole Foods still stocks many ultra-processed foods, but its catalog is less dominated by them than Target's or Walmart's",
        fontsize=21,
        fontweight="bold",
        color="#1f1f1f",
        ha="left",
        va="top",
    )
    title_ax.text(
        0.0,
        0.62,
        "Across 26,250 grocery items, ultra-processed products make up most of every store's assortment. The biggest Whole Foods advantage appears in a few ingredient-adjacent categories, not in center-aisle staples where processing is nearly universal.",
        fontsize=11.5,
        color="#4a4a4a",
        ha="left",
        va="top",
        wrap=True,
    )

    legend_x = 0.0
    for idx, cls in enumerate([0.0, 1.0, 2.0, 3.0]):
        x = legend_x + idx * 0.25
        title_ax.add_patch(
            Rectangle(
                (x, 0.10),
                0.02,
                0.09,
                transform=title_ax.transAxes,
                facecolor=CLASS_COLORS[cls],
                edgecolor="none",
            )
        )
        title_ax.text(
            x + 0.028,
            0.145,
            CLASS_LABELS[cls],
            transform=title_ax.transAxes,
            fontsize=10,
            color="#3b3b3b",
            ha="left",
            va="center",
        )

    draw_stack(left_ax, mix)
    left_ax.set_title("Overall product mix", loc="left", fontsize=13, pad=12, fontweight="bold")
    left_ax.text(
        0.0,
        -0.12,
        "Whole Foods has roughly twice the minimally processed share of Target.",
        transform=left_ax.transAxes,
        fontsize=9.5,
        color="#555555",
        ha="left",
    )

    draw_dumbbell(right_ax, category_summary)
    right_ax.set_title(
        "Where Whole Foods pulls away",
        loc="left",
        fontsize=13,
        pad=12,
        fontweight="bold",
    )
    right_ax.text(
        0.0,
        -0.12,
        "Shown: categories stocked by all three stores with at least 40 items each.",
        transform=right_ax.transAxes,
        fontsize=9.5,
        color="#555555",
        ha="left",
    )

    source_text = (
        "Source: grocerydb.csv provided in DSC 106 Project 1. "
        "Computed by grouping each store's catalog and each shared category by FPro_class."
    )
    fig.text(0.01, 0.03, source_text, fontsize=9.2, color="#666666")

    fig.savefig(OUT_DIR / "project1_visualization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rationale(df: pd.DataFrame) -> None:
    mix = build_overall_mix(df)
    category_summary = build_category_summary(df).sort_values("wf_advantage", ascending=False)

    top_gap = category_summary.iloc[0]
    second_gap = category_summary.iloc[1]

    paragraphs = [
        (
            "Guiding question: If you compared the online grocery catalogs of Target, Walmart, "
            "and Whole Foods, how different would their exposure to ultra-processed food really be?"
        ),
        (
            "The chart answers that question in two steps. The stacked bars establish the big-picture "
            "takeaway first: ultra-processed foods dominate all three catalogs, but Whole Foods is "
            f"meaningfully lower ({mix.loc['WholeFoods', 3.0]:.0%}) than Walmart ({mix.loc['Walmart', 3.0]:.0%}) "
            f"and Target ({mix.loc['Target', 3.0]:.0%}). I used proportions instead of raw counts so that the "
            "comparison is fair even though the stores contribute different numbers of products to the dataset."
        ),
        (
            "The dumbbell panel then narrows to categories carried by all three stores and highlights where "
            "Whole Foods differs most. Its largest gaps show up in categories such as "
            f"{CATEGORY_LABELS[top_gap['category']].lower()} and {CATEGORY_LABELS[second_gap['category']].lower()}, "
            "where Whole Foods carries noticeably fewer ultra-processed options. That second panel matters because "
            "the overall average could otherwise hide whether the difference is broad-based or concentrated in a few areas."
        ),
        (
            "I designed the piece as an expository visualization rather than an exploratory dashboard. The headline "
            "states the conclusion up front, the left panel gives immediate context, and the right panel explains "
            "where the pattern comes from. Warm red is reserved for ultra-processed food so the dominant category is "
            "visually obvious, while store colors stay consistent in the annotations and category comparison panel. "
            "I also kept labels and footnotes inside the figure so the visualization can stand on its own."
        ),
    ]

    wrapped = []
    for paragraph in paragraphs:
        wrapped.append(textwrap.fill(paragraph, width=102))

    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.08,
        0.95,
        "Project 1 Description + Design Rationale",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
        color="#1f1f1f",
    )
    ax.text(
        0.08,
        0.918,
        "Visualization title: Whole Foods still stocks many ultra-processed foods, but its catalog is less dominated by them than Target's or Walmart's",
        fontsize=10.5,
        ha="left",
        va="top",
        color="#444444",
        wrap=True,
    )

    y = 0.86
    for paragraph in wrapped:
        ax.text(0.08, y, paragraph, fontsize=11.2, ha="left", va="top", color="#222222")
        y -= 0.16

    fig.savefig(OUT_DIR / "project1_rationale.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    save_visualization(df)
    save_rationale(df)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
