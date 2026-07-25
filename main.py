import argparse
from datetime import datetime
import os
import json
import time

import pandas as pd
import requests



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments for a given variation."
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="Structured",
        help="Name of the variation folder inside 'UseCases/'."
    )
    parser.add_argument(
        "--usecase_folder",
        type=str,
        default="UseCases",
        help="Path to the folder containing the use cases"
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="Results",
        help="Path to the folder containing results"
    )
    return parser.parse_args()


# -----------------------------
# CONFIGURATION
# -----------------------------
wait_between_requests = 5  # seconds

# Set your key here or pass via environment variable
OPENROUTER_API_KEY = ""


# Models to test
MODELS = [  "deepseek/deepseek-v3.2-exp",
            "google/gemini-3-pro-preview",
            "meta-llama/llama-4-maverick",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "openai/gpt-oss-120b",
            "qwen/qwen3-vl-30b-a3b-thinking",
            "x-ai/grok-4.1-fast"
          ]

HYPERS = [
    {"temperature": 0, "top_p": 1},
    {"temperature": 0.2, "top_p": 1},
    {"temperature": 0.4, "top_p": 1},
    {"temperature": 0.6, "top_p": 1},
    {"temperature": 0.8, "top_p": 1},
    {"temperature": 1.0, "top_p": 1}
]


# -----------------------------
# FUNCTION TO QUERY OPENROUTER
# -----------------------------
def query_model(model, prompt, temperature, top_p):
    retry_delay = 5
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": 6500,
                    'provider': {
                        'require_parameters': True,
                    },
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Health",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "decision": {
                                        "type": "string",
                                        "description": "YES if one or more violations occurred; NO if none occurred"
                                    },
                                    "number-violations": {
                                        "type": "number",
                                        "description": "The number of violations found"
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Explanation of the reasoning that led to your decision"
                                    },
                                    "patient-explanation": {
                                        "type": "string",
                                        "description": "Explanation provided to the patient"
                                    },
                                    "physician-explanation": {
                                        "type": "string",
                                        "description": "Explanation provided to the physician"
                                    }
                                },
                                "required": ["decision", "number-violations", "reasoning", "patient-explanation", "physician-explanation"],
                                "additionalProperties": False
                            },
                        },
                    },
                }
            )

            if response.status_code == 429:
                print(f" Rate limit hit. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            return f"ERROR: {str(e)}"

        except (KeyError, IndexError):
            return f"ERROR: Unexpected response format: {response.text}"


# -----------------------------
# SAVE RESULTS
# -----------------------------
results = []

def save_result(usecase, file_name, model, temperature, top_p, content):
    results.append({
        "usecase": usecase,
        "file": file_name,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "output": content,
        "timestamp": datetime.now().isoformat()
    })



def main():
    args = parse_args()

    VARIATION = args.variation
    USECASE_FOLDER = args.usecase_folder + f"/{VARIATION}"
    OUTPUT_FOLDER = args.results_folder + f"/{VARIATION}"

    print(f"PROCESSING USECASES FOLDER {USECASE_FOLDER}")

    if not os.path.exists(USECASE_FOLDER):
        print(f"ERROR: The variation folder '{USECASE_FOLDER}' does not exist.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for folder in os.listdir(USECASE_FOLDER):
        usecase_path = os.path.join(USECASE_FOLDER, folder)
        if not os.path.isdir(usecase_path):
            continue

        usecase_results = []

        for file in os.listdir(usecase_path):
            if not file.endswith(".txt"):
                continue

            problem_file = os.path.join(usecase_path, file)
            with open(problem_file, "r", encoding="utf-8") as f:
                prompt = f.read()

            print(f"\n=== Processing {folder}/{file} ===")

            case_output = os.path.join(OUTPUT_FOLDER, folder)
            os.makedirs(case_output, exist_ok=True)

            for model in MODELS:
                for hyper in HYPERS:
                    temp = hyper["temperature"]
                    top_p = hyper["top_p"]
                    print(f"-> Model {model} | T={temp} | top_p={top_p}")

                    result = query_model(model, prompt, temp, top_p)

                    save_result(folder, file, model, temp, top_p, result)

                    usecase_results.append({
                        "usecase": folder,
                        "file": file,
                        "model": model,
                        "temperature": temp,
                        "top_p": top_p,
                        "output": result,
                        "timestamp": datetime.now().isoformat()
                    })

                    owner, model_op = model.split("/")
                    # save output files
                    os.makedirs(os.path.join(case_output, owner), exist_ok=True)

                    filename = f"{model_op.replace(':', '_')}_T{temp}_P{top_p}.json"
                    save_path = os.path.join(case_output, owner, filename)

                    # If result is already JSON text, try to parse it; if not, save raw text safely.
                    try:
                        parsed = json.loads(result)
                    except Exception:
                        parsed = {"raw_output": result}

                    with open(save_path, "w", encoding="utf-8") as out:
                        json.dump(parsed, out, indent=4, ensure_ascii=False)

                    time.sleep(wait_between_requests)

        # Save CSV per usecase
        usecase_df = pd.DataFrame(usecase_results)
        usecase_csv_path = os.path.join(OUTPUT_FOLDER, f"{folder}_results.csv")
        usecase_df.to_csv(usecase_csv_path, index=False)
        print(f"Saved CSV for use case '{folder}' at {usecase_csv_path}")

    # Save global CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "all_results.csv"), index=False)
    print("\n All results saved to 'all_results.csv'!")


if __name__ == "__main__":
    main()
