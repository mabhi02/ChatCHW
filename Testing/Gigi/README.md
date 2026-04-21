# Gigi DMN Runner

This folder contains a practical DMN-side prototype for black-box testing.  
`product_dmn_runner.py` runs product-team JSON logic on patient inputs and emits structured execution logs.

## What `product_dmn_runner.py` does

- Loads product-team logic from `clinical_logic.json`
- Runs one or more patients through the router + module rules
- Produces per-patient structured logs with:
  - `timestamp`
  - `patient_id`
  - `inputs`
  - `module_results`
  - `final_outcome`
- Writes logs as a JSON list for comparison-readiness

## Supported Input Modes

`product_dmn_runner.py` supports three input modes via `--patients-source`:

- `built_in`  
  Uses hardcoded sample patients inside the script.

- `dmn_ready_json` (default)  
  Loads DMN-ready inputs from `dmn_ready_sample_patients.json` (or `--patients-path`).

- `angelina_output`  
  Loads Angelina's `output.json`, calls her `batch_preprocess(...)`, then runs those converted DMN-ready patients.

## Default Logic Artifact Location

By default, the runner expects:

- `Testing/Gigi/clinical_logic.json`

If your artifact is elsewhere, pass `--logic "<path/to/clinical_logic.json>"`.

## Exact Run Commands

Run from repo root (`ChatCHW`):

- Built-in sample mode:
  - `python3 Testing/Gigi/product_dmn_runner.py --patients-source built_in --logic "Testing/Gigi/clinical_logic.json"`

- DMN-ready JSON mode:
  - `python3 Testing/Gigi/product_dmn_runner.py --patients-source dmn_ready_json --patients-path "Testing/Gigi/dmn_ready_sample_patients.json" --logic "Testing/Gigi/clinical_logic.json"`

- Angelina preprocessor adapter mode:
  - `python3 Testing/Gigi/product_dmn_runner.py --patients-source angelina_output --angelina-output-path "Testing/Angelina/dmn/output.json" --angelina-preprocessor-path "Testing/Angelina/dmn/dmn_preprocessor.py" --logic "Testing/Gigi/clinical_logic.json"`

## Output Log Location

By default, logs are written to:

- `Testing/Gigi/product_dmn_sample_logs.json`

You can override with `--output "<path/to/logs.json>"`.

## Assumptions and Limitations

- The runner assumes product-team logic is available as JSON (`clinical_logic.json`) with `modules` and `router.rows`.
- Expression evaluation uses a lightweight Python translation of rule conditions and fails closed (`False`) on expression errors.
- The prototype prioritizes execution + structured logging, not medical correctness validation.
- `angelina_output` mode currently calls `batch_preprocess(...)` with default context only; if malaria/RDT/MUAC context is needed, add a small context-wiring extension.
- Output includes `execution_trace` as extra debug context beyond the core log schema fields.
