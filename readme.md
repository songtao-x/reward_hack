# Gradient & Reward-Hacking Pipeline (ARLSAT / Big-Math / Code)

This project contains the gradient-extraction and gradient-analysis pipeline used for reward-hacking related experiments across multiple datasets (e.g., **ARLSAT**, **Big-Math**, **Code**).

---

## Repository Structure 

### 1) Gradient Pipeline
Location: `arlsat/icl/gradient/`

This folder contains everything related to extracting gradients and running downstream gradient analysis.

#### Core files
- **`analysis.py`**
  - Main entry for **gradient analysis** (e.g., getting gradient, SVM, clustering, metrics).
  - If you want to modify analysis logic, start here.

- **`gradient_h.py`**
  - Implements the **gradient extraction** workflow. (including layer selection for gradient)
  - Calls functions from `gradient.py`.

- **`gradient.py`**
  - Lower-level functions/utilities used by `gradient_h.py`.
  - Typically contains shared helpers.

---

### 2) Dataset + Pipeline 
Location: `rh_model_setting.py`

This file is the main orchestration entry for:
- Dataset implementations and loading logic (e.g., **ARLSAT**, **Big-Math**, **Code**)
- Running the end-to-end pipeline that prepares data and triggers gradient extraction/analysis

**Position:** arlsat/pipeline/rh_model_setting.py, big_math/trace/rh_model_setting.py


start from **`rh_model_setting.py`**.

---

## High-Level Flow

1. **Dataset setup / configuration**
   - Implemented and managed in: `rh_model_setting.py`

2. **Gradient extraction**
   - Main code: `arlsat/icl/gradient/gradient_h.py`
   - Helpers: `arlsat/icl/gradient/gradient.py`

3. **Gradient analysis**
   - Main analysis logic: `arlsat/icl/gradient/analysis.py`

---

## Notes / Conventions

- `gradient_h.py` is the “pipeline-level” gradient extractor.
- `gradient.py` should stay modular and reusable across datasets.
- `analysis.py` should focus on analysis-only operations 
- Dataset-specific preprocessing belongs in `rh_model_setting.py` (or its imported dataset modules).

---
