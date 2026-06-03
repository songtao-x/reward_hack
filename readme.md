# Detecting and Suppressing Reward Hacking with Gradient Fingerprint

Official implementation for the paper **"Detecting and Suppressing Reward Hacking with Gradient Fingerprint"**.

This repository provides the gradient-extraction and gradient-analysis pipeline used in our reward-hacking experiments across multiple benchmarks, including **ARLSAT**, **Big-Math**, and **Code**.

---

## Overview

Reward hacking occurs when models exploit unintended shortcuts in a reward signal rather than learning the target behavior. In this work, we introduce a **gradient fingerprint** approach that:

1. Extracts per-example gradients from intermediate checkpoints of RL-trained models.
2. Analyzes these gradients (e.g., via SVM classification, clustering, and distributional metrics) to identify behaviors consistent with reward hacking.
3. Enables downstream **detection** and **suppression** of hacked behavior during training.

---

## Repository Structure

### 1. Gradient Pipeline

**Location:** `arlsat/icl/gradient/`

This folder contains the core modules for extracting and analyzing gradients.

| File | Description |
| --- | --- |
| `analysis.py` | Main entry point for **gradient analysis** (gradient aggregation, SVM classification, clustering, and evaluation metrics). Start here to modify analysis logic. |
| `gradient_h.py` | Implements the **gradient extraction** workflow, including layer selection. Calls into `gradient.py`. |
| `gradient.py` | Lower-level utilities and shared helpers used by `gradient_h.py`. |

### 2. Dataset and Pipeline Configuration

**Location:** `rh_model_setting.py` (per-dataset variant)

Each dataset has its own `rh_model_setting.py` that defines dataset loading, model checkpoints, and the full end-to-end pipeline (data preparation → gradient extraction → analysis).

---

## Supported Benchmarks

### Big-Math

**Core module:** `big_math/trace/rh_model_setting.py`

**Configuration**

- Dataset save directory (`save_dir`):
  ```
  /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/big_math/trace/data/rloo_cheat_step_{s}
  ```
  Gradient files are stored under two subfolders used by the gradient stage: `false_gradient/` and `true_gradient/`.

- Model checkpoint (`model_name`):
  ```
  xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}
  ```
  where `s ∈ range(10, 45, 5)`.

**Quick start**

Run the full pipeline:
```bash
cd big_math
python -m trace.main_bigmath
```

Run gradient analysis only:
```bash
cd big_math
python -m trace.main_bigmath --gradient_only
```

---

### ARLSAT

**Core module:** `arlsat/pipeline/rh_model_setting.py`

**Configuration**

- Dataset save directory (`save_dir`):
  ```
  /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}
  ```
  Gradient files are stored under `false_gradient/` and `true_gradient/`.

- Model checkpoint (`model_name`):
  ```
  /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf
  ```
  where `s ∈ range(10, 45, 5)`.

**Quick start**

Run the full pipeline:
```bash
python -m arlsat.pipeline.main_q3_pro
```

Run gradient analysis only:
```bash
python -m arlsat.pipeline.main_q3_pro --gradient_only
```

---

## High-Level Flow

1. **Dataset setup and configuration** — implemented per-benchmark in `rh_model_setting.py`.
2. **Gradient extraction** — driven by `arlsat/icl/gradient/gradient_h.py`, with helpers in `arlsat/icl/gradient/gradient.py`.
3. **Gradient analysis** — performed by `arlsat/icl/gradient/analysis.py`, producing the gradient fingerprints used for detection and suppression.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026detectingsuppressingrewardhacking,
      title={Detecting and Suppressing Reward Hacking with Gradient Fingerprints}, 
      author={Songtao Wang and Quang Hieu Pham and Fangcong Yin and Xinpeng Wang and Jocelyn Qiaochu Chen and Greg Durrett and Xi Ye},
      year={2026},
      eprint={2604.16242},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.16242}, 
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
