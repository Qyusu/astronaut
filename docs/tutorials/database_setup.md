# Database Setup

## 1. (Optional) Prepare PennyLane Documentation

Clone the [PennyLane repository](https://github.com/PennyLaneAI/pennylane) and check out the version you need (v0.39.0 was used in the paper). Astronaut converts the source into a JSON reference that is later consumed during code generation.

```bash
# run from astronaut/astronaut
python3 db/pennylane/setup_local_file.py --save_raw_json_path="../data/references/raw_v0.39.0.json" --save_formatted_json_path="../data/references/v0.39.0.json"
```

<details><summary>Available flags</summary>

- `--save_raw_json_path` (Path, required): Output path for the raw PennyLane reference.
- `--save_formatted_json_path` (Path, required): Output path for the formatted PennyLane reference.

</details>

## 2. (Optional) Prepare arXiv Papers

Astronaut can index scientific papers retrieved through the [arXiv API](https://info.arxiv.org/help/api/index.html). Please throttle requests to avoid stressing the public API. Retrieved PDFs are stored under `data/papers/arxiv` and should remain available for subsequent runs.

```bash
# run from astronaut/astronaut
python3 db/setup_arxiv_db.py --index_name="arxiv-v2" --chunk_size=1024 --max_results_per_request=100 --init_db "true" --category="quant-ph" --keywords="Quantum Machine Learning, Variational Quantum Eigensolver, VQE, Variational Eigenvalue Solver, Quantum Chemistry" --date_range="20200101-20250430"
```

<details><summary>Available flags</summary>

- `--chunk_size` (integer, required): Text chunk size used during ingestion. Default: 1024.
- `--max_results_per_request` (integer, required): Maximum number of papers retrieved per API request. Default: 100.
- `--init_db` (boolean, required): Initialize the database (`true`) or append (`false`).
- `--category` (string, required): arXiv category (e.g., `"quant-ph"`).
- `--keywords` (string, required): Comma-separated keywords used to filter papers.
- `--date_range` (string, required): Date range in the format `YYYYMMDD-YYYYMMDD`.

</details>
