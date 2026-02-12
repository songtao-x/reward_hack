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

- Dataset implementations and loading logic (e.g., **ARLSAT**, **Big-Math**, **Code**)
- Running the whole pipeline that prepares data and perform gradient extraction/analysis

#### Big-Math: 
**core code:**
- `big_math/trace/rh_model_setting.py`

**Settings**
- dataset save_dir (`save_dir`):
  - `/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/big_math/trace/data/rloo_cheat_step_{s}`
  - The gradient file saved under the folder: `false_gradient`, `true_gradient`. During gradient process, only these two files will be called 

- Model name / checkpoint (`model_name`):
  - `xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}`
`s` in range(10, 45, 5)

**Quick start**
- Run the full pipeline:
  ```bash
  python -m big_math.trace.main_big_math
  ```
- Run **gradient** test only:
  ```bash
  python -m big_math.trace.main_big_math --gradient_only
  ```


#### ARLSAT: 
**core code:**
- `arlsat/pipeline/rh_model_setting.py`

**Settings**
- dataset save_dir:
  - `/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}`
  - The gradient file saved under the folder: `false_gradient`, `true_gradient`. During gradient process, only these two files will be called 

- model_name:
  - `/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf`
`s` in range(10, 45, 5)

**Quick start**
- Run the full pipeline:
  ```bash
  python -m arlsat.pipeline.main_q3_pro
  ```
- Run **gradient** test only:
  ```bash
  python -m arlsat/pipeline/main_q3_pro --gradient_only
  ```


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

