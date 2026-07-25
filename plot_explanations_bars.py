import ast
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


def parse_args():
    parser = argparse.ArgumentParser(description="Compare structured vs unstructured distances")
    parser.add_argument("--csv_structured", type=str, required=True)
    parser.add_argument("--csv_unstructured", type=str, required=True)
    parser.add_argument("--explanation_type", type=str, default="patient-explanation",
                        choices=["patient-explanation", "physician-explanation"])
    parser.add_argument("--output_folder", type=str, default="plots")
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine"])
    return parser.parse_args()


def extract_explanations(df, explanation_type):
    out = []
    for val in df["output"]:
        D = ast.literal_eval(val)
        out.append(D.get(explanation_type, ""))
    return out


def compute_dist(vecs, centroid, metric):
    if metric == "cosine":
        return cosine_distances(vecs, centroid).mean()
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    dfS = pd.read_csv(args.csv_structured)
    dfU = pd.read_csv(args.csv_unstructured)

    model_encoder = SentenceTransformer("all-mpnet-base-v2")

    usecases = sorted(set(dfS.usecase).intersection(dfU.usecase))

    for usecase in usecases:
        dfS_uc = dfS[dfS.usecase == usecase]
        dfU_uc = dfU[dfU.usecase == usecase]

        temps = sorted(set(dfS_uc.temperature).intersection(dfU_uc.temperature))
        all_models = sorted(set(dfS_uc.model).union(dfU_uc.model))

        # Store results for plotting
        records = []

        for temp in temps:
            dfS_temp = dfS_uc[dfS_uc.temperature == temp]
            dfU_temp = dfU_uc[dfU_uc.temperature == temp]

            exp_S = extract_explanations(dfS_temp, args.explanation_type)
            exp_U = extract_explanations(dfU_temp, args.explanation_type)

            vecs_S = model_encoder.encode(exp_S)
            vecs_U = model_encoder.encode(exp_U)

            centroid_S = np.mean(vecs_S, axis=0, keepdims=True)
            centroid_U = np.mean(vecs_U, axis=0, keepdims=True)

            for mdl in all_models:
                exp_s_m = extract_explanations(dfS_temp[dfS_temp.model == mdl], args.explanation_type)
                exp_u_m = extract_explanations(dfU_temp[dfU_temp.model == mdl], args.explanation_type)

                if len(exp_s_m) == 0 or len(exp_u_m) == 0:
                    continue

                vec_s_m = model_encoder.encode(exp_s_m)
                vec_u_m = model_encoder.encode(exp_u_m)

                dist_S = compute_dist(vec_s_m, centroid_S, args.distance)
                dist_U = compute_dist(vec_u_m, centroid_U, args.distance)

                records.append([temp, mdl, dist_S, dist_U])

        plot_df = pd.DataFrame(records, columns=["temperature", "model", "dist_S", "dist_U"])

        # --- Plot ---
        plt.figure(figsize=(14, 6))

        temps = sorted(plot_df.temperature.unique())
        n_models = len(all_models)
        n_temps = len(temps)

        # Width for each bar
        total_width = 0.8  # total width for one temperature group
        bar_width = total_width / (n_models * 2)  # each model has 2 bars

        # colormap for models
        cmap = plt.colormaps["tab10"]
        model_colors = {mdl: cmap(i % 10) for i, mdl in enumerate(all_models)}

        # X positions for each temperature
        x_base = np.arange(n_temps)

        for i, mdl in enumerate(all_models):
            df_m = plot_df[plot_df.model == mdl]
            # Compute offsets for bars: each model has 2 bars side by side
            offset = (-total_width/2) + i*2*bar_width

            # Structured bar (solid)
            plt.bar(x_base + offset, df_m["dist_S"], width=bar_width, color=model_colors[mdl], label=mdl if i==0 else None)

            # Unstructured bar (hatch)
            plt.bar(x_base + offset + bar_width, df_m["dist_U"], width=bar_width, color=model_colors[mdl], hatch='//', alpha=0.7)

        plt.xticks(x_base, temps)
        plt.ylim(0, 1)  # set y-axis limits from 0 to 1
        plt.yticks(np.arange(0, 1.1, 0.1))  # set ticks every 0.1
        plt.xlabel("Temperature")
        plt.ylabel(f"{args.distance.title()} Distance")
        plt.title(f"Structured vs Unstructured Explanation Distances\nUsecase: {usecase} ({args.explanation_type})")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Legend - V1
        # handles = [plt.Rectangle((0,0),1,1,color=model_colors[m], label=m) for m in all_models]
        # handles.append(plt.Rectangle((0,0),1,1,color='grey', hatch='//', alpha=0.7, label='Unstructured'))
        # handles.append(plt.Rectangle((0,0),1,1,color='grey', label='Structured'))

        # plt.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
        # Legend: per-model structured (solid) and unstructured (hatched) entries

        # Legend - V2
        handles = []
        for mdl in all_models:
            color = model_colors[mdl]
            # Structured legend entry
            handles.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f"{mdl} (ST)")
            )
            # Unstructured legend entry
            handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', hatch='//', alpha=0.7,
                              label=f"{mdl} (UNST)")
            )


        # To Comment if we want to get plots without legend
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        savepath = os.path.join(args.output_folder, f"{usecase}_{args.explanation_type}_comparison_bars.png")
        plt.savefig(savepath)
        plt.close()
        print(f"Saved combined temperature plot for {usecase}: {savepath}")


if __name__ == "__main__":
    main()
