"""Week 2 — Exploratory data analysis for WDBC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import load_wdbc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "eda"


def run_eda() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_wdbc()
    feature_cols = [c for c in df.columns if c != "diagnosis"]

    summary = {
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "benign": int((df["diagnosis"] == 0).sum()),
        "malignant": int((df["diagnosis"] == 1).sum()),
        "malignant_rate": float((df["diagnosis"] == 1).mean()),
        "missing_values": int(df.isna().sum().sum()),
    }
    pd.Series(summary).to_csv(OUT / "summary.csv", header=["value"])
    df[feature_cols].describe().T.to_csv(OUT / "feature_describe.csv")

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["diagnosis"].map({0: "Benign", 1: "Malignant"}).value_counts()
    counts.plot(kind="bar", color=["#1f7a4d", "#a33b3b"], ax=ax, rot=0)
    ax.set_title("WDBC class balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap (mean features only for readability)
    mean_cols = [c for c in feature_cols if c.endswith("_mean")]
    corr = df[mean_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Correlation — mean features")
    fig.tight_layout()
    fig.savefig(OUT / "corr_mean_features.png", dpi=150)
    plt.close(fig)

    # Diagnosis vs a few key features
    key = [
        c
        for c in [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "concave_points_mean",
            "concavity_mean",
        ]
        if c in df.columns
    ]
    melted = df.melt(
        id_vars=["diagnosis"],
        value_vars=key,
        var_name="feature",
        value_name="value",
    )
    melted["diagnosis"] = melted["diagnosis"].map({0: "Benign", 1: "Malignant"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="diagnosis", ax=ax)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Key feature distributions by diagnosis")
    fig.tight_layout()
    fig.savefig(OUT / "key_feature_boxplots.png", dpi=150)
    plt.close(fig)

    print("EDA summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved plots → {OUT}")


if __name__ == "__main__":
    run_eda()
