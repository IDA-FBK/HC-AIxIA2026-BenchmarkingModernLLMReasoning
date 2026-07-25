# HC-AIxIA2026-BenchmarkinModernLLMTemporalReasoning
Repository for the paper "Benchmarking Temporal Reasoning and Audience-Aware Explanation Generation of Modern Large Language Models on Clinical Data"

---

## Repository Content

- **Results/**
  
  Contains `Structured` and `Unstructured` folders. Each folder includes CSV files with aggregated responses for:  
  - Single use cases  
  - All use cases combined  

- **Plots/**
   
  Contains visualizations:  
  - **Decision plots:** LLM decisions for both structured and unstructured cases  
  - **Explanation plots:** Explanation distances plotted as bars or points. Two versions exist: one with legends and one without.  

- **Scripts:**
   
  - `main.py` : Runs the main experiment  
  - `plot_decision.py` : Plots LLM decisions from CSV results  
  - `plot_explanations_bars.py` : Plots explanation distances as bars
  - `plot_explanations_points.py` : Plots explanation distances as points

- **Poetry files:**
   
  - `poetry.lock` : Locks dependency versions  
  - `pyproject.toml` : Project configuration and dependencies  

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
```

### 2. Download Poetry

[Poetry website](https://python-poetry.org/)

### 3. Install project dependencies
```bash
cd LLM-Benchmarking-Temporal-Clinical-Reasoning
poetry install --no-root
```

### 4. Activate the virtual environment

**Windows:**  
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux:**  
```bash
source .venv/bin/activate
```

---

## Running Experiments

### 1. Set your OpenRouter API key

In `main.py`:
```python
OPENROUTER_API_KEY = "<your-api-key>"
```

### 2. Run the main experiment
```bash
python main.py --variation Structured --usecase_folder UseCases --results_folder Results
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--variation` | Name of the variation folder inside `UseCases/` | `Structured` |
| `--usecase_folder` | Path to the folder containing the use cases | `UseCases` |
| `--results_folder` | Path to the folder to store results | `Results` |

---

## Aggregating Results

### 1. Plot LLM decisions
```bash
python plot_decision.py --csv <results_csv> --variation ST
```

Example:
```bash
python .\plot_decisions.py --csv .\Results\LLMResponses\Structured\CSV\USECASE1-NoViolation_results.csv
 ```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv` | Path to the results CSV file | Required |
| `--variation` | Variation to compute plots from (`ST` or `UNST`) | `ST` |

### 2. (Additional) Plot Explanation Distances (Bars)
```bash
python plot_explanations_bars.py \
    --csv_structured <structured_csv> \
    --csv_unstructured <unstructured_csv> \
    --explanation_type patient-explanation \
    --output_folder plots \
    --distance cosine
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv_structured` | Structured CSV file | Required |
| `--csv_unstructured` | Unstructured CSV file | Required |
| `--explanation_type` | `patient-explanation` or `physician-explanation` | `patient-explanation` |
| `--output_folder` | Folder to save the plots | `plots` |
| `--distance` | Distance metric to compute| `cosine` |

Example:
```bash
python .\plot_explanations_bars.py --csv_structured .\Results\LLMResponses\Structured\CSV\USECASE1-NoViolation_results.csv --csv_unstructured .\Results\LLMResponses\Unstructured\CSV\USECASE1-NoViolation_results.csv --explanation_type "physician-explanation"
 ```


### 3. (Additional) Plot Explanation Distances (Point)
```bash
python plot_explanations_bars.py \
    --csv_file <csv_results_file> \
    --explanation_type patient-explanation \
    --output_folder plots \
    --distance cosine \
    --variation ST
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv_file` | (Structured or unstructured) CSV file | Required |
| `--explanation_type` | `patient-explanation` or `physician-explanation` | `patient-explanation` |
| `--output_folder` | Folder to save the plots | `plots` |
| `--distance` | Distance metric to compute| `cosine` |
| `--variation` | Strcutured (ST) or Unstructured (UNST)| `ST` |


Example:
```bash
 python .\plot_explanations_points.py --csv_file .\Results\LLMResponses\Structured\CSV\USECASE1-NoViolation_results.csv  --explanation_type "physician-explanation"
 ```




---

## Notes

- Make sure the virtual environment is active before running any scripts.  
- All CSV results will be saved in the `Results/` folder by default.  
- Replace `<repository-url>` with the URL of this GitHub repository.

## Authors
- Gianluca Apriceno: apriceno@fbk.eu
- Tania Bailoni: tbailoni@fbk.eu
- Mauro Dragoni: dragoni@fbk.eu

