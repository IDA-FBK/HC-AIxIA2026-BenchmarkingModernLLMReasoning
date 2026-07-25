import ast
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Plot explanation distances per usecase and temperature")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file with results")
    parser.add_argument("--explanation_type", type=str, default="patient-explanation",
                        choices=["patient-explanation", "physician-explanation"],
                        help="Which explanation to use from the 'output' column")
    parser.add_argument("--output_folder", type=str, default="plots", help="Folder to save plots")
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine"],
                        help="Distance metric")
    parser.add_argument("--variation", type=str, default="ST", choices=["ST", "UNST"],
                        help="Variation to compute plots from")
    return parser.parse_args()


def extract_explanations(df_temp, explanation_type):
    """Extract the chosen explanation from the 'output' column (Python dict strings)"""
    explanations = []
    for val in df_temp['output']:
        obj = ast.literal_eval(val)  # safely parse Python dict string
        explanations.append(obj.get(explanation_type, ""))  # get explanation or empty string
    return explanations


def compute_distance_matrix(vectors, metric="cosine"):
    if metric == "cosine":
        return cosine_distances(vectors)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def main():
    args = parse_args()
    variation = args.variation
    explanation_type = args.explanation_type
    df = pd.read_csv(args.csv_file)
    os.makedirs(args.output_folder, exist_ok=True)


    # Initialize sentence transformer
    model = SentenceTransformer('all-mpnet-base-v2')#('all-MiniLM-L6-v2')

    usecases = df['usecase'].unique()

    for usecase in usecases:
        df_uc = df[df['usecase'] == usecase]
        temps = sorted(df_uc['temperature'].unique())
        distances_data = []

        for temp in temps:
            df_temp = df_uc[df_uc['temperature'] == temp]

            # Extract explanations
            explanations = extract_explanations(df_temp, explanation_type)

            # Compute embeddings
            vectors = model.encode(explanations)
            avg_vector = np.mean(vectors, axis=0, keepdims=True)
            # Compute distance to centroid (average vector)
            if args.distance == "cosine":
                dist = cosine_distances(vectors, avg_vector).flatten()
            else:
                dist = euclidean_distances(vectors, avg_vector).flatten()

            distances_data.extend(list(zip(df_temp['model'], [temp] * len(dist), dist)))

        # Prepare data for plotting
        plot_df = pd.DataFrame(distances_data, columns=["model", "temperature", "distance"])

        # Scatter plot for each model
        plt.figure(figsize=(8, 6))
        for model_name, grp in plot_df.groupby("model"):
            plt.scatter(grp["temperature"], grp["distance"], label=model_name, alpha=0.6)

        plt.title(f"Distance to average {explanation_type} vector\nUsecase ({variation}): {usecase}")
        plt.xlabel("Temperature")
        plt.ylabel(f"{args.distance.title()} Distance")
        plt.ylim(0, 1)  # set y-axis limits from 0 to 1
        plt.yticks(np.arange(0, 1.1, 0.1))  # set ticks every 0.1
        # Fixed, non-overlapping legend placement

        # To Comment if we want to get plots without legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(args.output_folder, f"{usecase}_{variation}_{explanation_type}_distance_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for usecase '{usecase}' at {save_path}")


if __name__ == "__main__":
    main()
