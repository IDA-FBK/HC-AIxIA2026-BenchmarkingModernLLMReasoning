import argparse
import ast
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process results CSV and generate heatmaps for violations/decisions."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the results CSV file"
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="ST",
        choices=["ST", "UNST"],
        help="Variation to compute plots from"
    )
    return parser.parse_args()


# -----------------------------
# Extract fields from Python dict-like strings
# -----------------------------
def extract_field(text, field):
    if pd.isna(text):
        return None
    try:
        data = ast.literal_eval(text)
        return data.get(field)
    except Exception:
        return None


# -----------------------------
# Normalize decision values
# -----------------------------
def normalize_decision(val, valid_decisions={"YES", "NO"}):
    if val in valid_decisions:
        return val
    return "OTH"


# -----------------------------
# Compute match status
# -----------------------------
def compute_status(row, expected_values):
    exp = expected_values.get(row["usecase"], {"violations": None, "decision": None})
    expected_decision = exp["decision"]
    expected_violation = exp["violations"]

    if row["decision_cleaned"] == "OTH":
        return np.nan  # Use NaN to mark UNK separately

    if pd.notna(row["predicted_violations"]):
        return (row["predicted_violations"] == expected_violation) and \
               (row["decision_cleaned"] == expected_decision)
    else:
        return row["decision_cleaned"] == expected_decision


# -----------------------------
# Create annotation for heatmap
# -----------------------------
def make_annotation(row):
    # If decision is unknown, still include the number of violations if available
    if row["decision_cleaned"] == "UNK":
        if pd.notna(row["predicted_violations"]):
            return f'UNK/{int(row["predicted_violations"])}'
        else:
            return "UNK"
    # Normal YES/NO cases
    if pd.notna(row["predicted_violations"]):
        return f'{row["decision_cleaned"]}/{int(row["predicted_violations"])}'
    else:
        return row["decision_cleaned"]


# -----------------------------
# Map status to index for custom colormap
# -----------------------------
def map_status_for_cmap(val):
    if pd.isna(val):  # UNK
        return 2
    elif val:        # match
        return 1
    else:            # mismatch
        return 0


# -----------------------------
# Generate heatmap
# -----------------------------
def generate_heatmap(df, usecase, expected_values, variation, output_dir="heatmaps"):
    # Save unexpected decisions
    subset_unk = df[(df["usecase"] == usecase) & (df["decision_cleaned"] == "OTH")]
    if len(subset_unk) > 0:
        filename = f"unexpected_decisions_{usecase}_{variation}.csv"
        subset_unk.to_csv(filename, index=False)
        print(f"Saved unexpected decisions for {usecase} to {filename}")

    subset = df[df["usecase"] == usecase]

    pivot_colors = subset.pivot(index="model", columns="temperature", values="status")
    # Safe mapping for colormap
    pivot_colors = pivot_colors.apply(lambda col: col.map(map_status_for_cmap))

    pivot_annot  = subset.pivot(index="model", columns="temperature", values="annotation")

    exp = expected_values.get(usecase, {"violations": "?", "decision": "?"})

    # Custom colormap: red=mismatch, green=match, orange=UNK
    cmap = ListedColormap(["red", "green", "orange"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_colors,
        annot=pivot_annot,
        fmt='',
        cmap=cmap,
        cbar=False,
        vmin=0, vmax=2  # Ensure exact mapping
    )

    plt.title(f"Use Case {usecase} ({variation}): D={exp['decision']} -- NV={exp['violations']}", fontsize=14)
    plt.xlabel("Temperature")
    plt.ylabel("Model")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/usecase_{usecase}_{variation}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")



def main():
    args = parse_args()
    csv_path = args.csv
    variation = args.variation

    if not os.path.isfile(csv_path):
        print(f"Error: File not found → {csv_path}")
        return

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    expected_values = {
        "USECASE1-NoViolation": {"violations": 0, "decision": "NO"},
        "USECASE2-SingleMealViolation": {"violations": 1, "decision": "YES"},
        "USECASE3-MultipleMealViolation": {"violations": 2, "decision": "YES"},
        "USECASE4-SingleWeeklyViolation": {"violations": 1, "decision": "YES"},
        "USECASE5-MultipleWeeklyViolation": {"violations": 2, "decision": "YES"},
    }

    # Extract fields
    df["predicted_violations"] = df["output"].apply(lambda x: extract_field(x, "number-violations"))
    df["decision"] = df["output"].apply(lambda x: extract_field(x, "decision"))
    df["predicted_violations"] = pd.to_numeric(df["predicted_violations"], errors="coerce")

    # Clean decisions
    df["decision_cleaned"] = df["decision"].apply(normalize_decision)

    # Compute status
    df["status"] = df.apply(lambda row: compute_status(row, expected_values), axis=1)

    # Annotation
    df["annotation"] = df.apply(make_annotation, axis=1)

    # Generate heatmaps per usecase
    usecases = df["usecase"].unique()
    for uc in usecases:
        generate_heatmap(df, uc, expected_values, variation)


if __name__ == "__main__":
    main()
