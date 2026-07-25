import argparse
import os
import json

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model JSON results into per-usecase and global CSVs."
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Path to the main Results/<Variation> folder"
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"raw_output": f.read()}


def parse_filename(filename):
    """
    Extract model, temperature, top_p from filenames like:
    meta-llama_llama-4-maverick_T0.6_P1.json
    """
    base = filename.replace(".json", "")
    parts = base.split("_")

    # last two parts must be T{temp} and P{top_p}
    temp = parts[-2].replace("T", "")
    top_p = parts[-1].replace("P", "")

    # the rest is model
    model = "_".join(parts[:-2]).replace("_", "/")  # reconstruct model name

    return model, float(temp), float(top_p)


def clean_model_name(model_name):
    """
    Normalize model names by removing known suffixes like '-preview' or ':free'.
    """
    # Remove trailing '-preview'
    if model_name.endswith("-preview"):
        model_name = model_name.rsplit("-preview", 1)[0]

    # Remove anything after '/' (like '/free')
    if "/" in model_name:
        model_name = model_name.split("/", 1)[0]

    return model_name


def main():
    args = parse_args()

    ROOT = args.results_folder
    print(f"Aggregating JSON results from: {ROOT}")

    global_results = []

    # Each folder inside ROOT is a usecase folder
    for usecase in os.listdir(ROOT):
        usecase_path = os.path.join(ROOT, usecase)
        if not os.path.isdir(usecase_path):
            continue

        print(f"\nProcessing usecase: {usecase}")
        usecase_entries = []

        # Each JSON belongs to some model config
        for root, dirs, files in os.walk(usecase_path):
            for file in files:
                if not file.endswith(".json"):
                    continue

                json_path = os.path.join(root, file)
                json_data = load_json(json_path)

                model, temp, top_p = parse_filename(file)
                model_clean = clean_model_name(model)

                entry = {
                    "usecase": usecase,
                    "model": model_clean,
                    "temperature": temp,
                    "top_p": top_p,
                    "output": json_data,
                }

                usecase_entries.append(entry)
                global_results.append(entry)

        # Write per-usecase CSV
        if usecase_entries:
            df_usecase = pd.DataFrame(usecase_entries)
            out_csv = os.path.join(ROOT, f"{usecase}_results.csv")
            df_usecase.to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")
        else:
            print(f"(No JSON files found in {usecase_path})")

    # Write global CSV
    if global_results:
        df_global = pd.DataFrame(global_results)
        out_csv = os.path.join(ROOT, "all_results.csv")
        df_global.to_csv(out_csv, index=False)
        print(f"\nSaved global CSV: {out_csv}")
    else:
        print("\nNo results found at all — ensure JSON files exist.")


if __name__ == "__main__":
    main()
