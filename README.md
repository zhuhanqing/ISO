<div align="center">

# <img src="assets/iso_symbol.svg" alt="ISO" height="34"/> ISO: An RLVR-Native Optimization Stack

<p align="center">
    <a href="https://zhuhanqing.github.io/">Hanqing Zhu</a><sup>1*</sup>,
    <a href="https://www.wenyancong.com/">Wenyan Cong</a><sup>1*</sup>,
    <a href="https://jamessand.github.io/">Zhizhou Sha</a><sup>1</sup>,
    <a href="https://sagnikmukherjee.github.io/">Sagnik Mukherjee</a><sup>2</sup>,
    <a href="https://www.researchgate.net/profile/Xinyuan-Song-6">Xinyuan Song</a><sup>3</sup>,
    <a href="https://scholar.google.com/citations?user=ftQvY4UAAAAJ">David González-Martínez</a><sup>6</sup>,<br>
    <a href="https://xwushirley.github.io/">Xiaoxia Wu</a><sup>4</sup>,
    <a href="https://yuandong-tian.com/">Yuandong Tian</a><sup>5</sup>,
    <a href="https://shiweiliuiiiiiii.github.io/">Shiwei Liu</a><sup>6</sup>,
    <a href="https://users.ece.utexas.edu/~dpan/">David Z. Pan</a><sup>1</sup>,
    <a href="https://vita-group.github.io/">Zhangyang "Atlas" Wang</a><sup>1†</sup>
</p>

<p align="center">
    <sup>1</sup>The University of Texas at Austin &nbsp; <sup>2</sup>UIUC &nbsp; <sup>3</sup>Emory University &nbsp; <sup>4</sup>Together AI<br>
    <sup>5</sup>Recursive Superintelligence Inc &nbsp; <sup>6</sup>ELLIS Institute Tübingen
</p>

<p align="center">
    <sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author
</p>

[![Project](https://img.shields.io/badge/🌐%20Project-Page-green)](https://iso-rlvr.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.19331-b31b1b.svg)](https://arxiv.org/abs/2607.19331)

***Inherit the spectrum, optimize the frames.***

</div>

Official repository for **ISO (Isospectral Optimization)**, an RLVR-native, fixed-spectrum optimization framework with complementary offline and online instantiations:

- **ISO-Merger** (offline): composes shared-base RLVR specialists directly from their checkpoints — no post-merge data, rollouts, gradient updates, or on-policy distillation.
- **ISO-Optimizer** (online): applies a conventional base optimizer (AdamW, Muon, ...) to the singular-frame variables `(U, V)` while keeping the base spectrum `Σ₀` fixed throughout RLVR training.

## 💡 Key Idea: Spectral Inheritance

Reinforcement learning with verifiable rewards (RLVR) is rapidly advancing the reasoning capabilities of language models, yet the optimization layer that converts reward feedback into weight-space updates remains poorly understood. Studying this layer through the singular structure of model weights (`W = U Σ Vᵀ`), we identify **spectral inheritance**:

> RLVR can **reuse the base model's weight spectra** `Σ₀` while acquiring new behavior through changes in the associated input and output **singular frames** `(U, V)`.

Concretely, we show that:

1. **Spectra stay.** Unconstrained RLVR checkpoints remain close to the fixed-spectrum family of their base weights (spectral residual ≈ 3% of the full checkpoint displacement), in sharp contrast to SFT.
2. **The spectral change is not needed.** Restoring the base spectrum after RLVR preserves most acquired gains, and keeping `Σ₀` fixed throughout training still supports strong learning — while a spectrum-only control (frames frozen) does not.
3. **Both frames must remain adaptable.** Remixing only within the incoming subspaces, or freezing either incoming singular subspace, leaves substantially more of the checkpoint update unexplained (median 87% / 45% / 42%) than fixing only the spectrum (median 1.8%).

ISO turns this regularity into a practical inductive bias for the RLVR post-training stack: represent post-training change in the fixed-spectrum parameterization `W(U, V) = U Σ₀ Vᵀ` and optimize / compose only the frames.

## 📈 Highlights

**ISO-Optimizer** — across math reasoning (Qwen3-1.7B/4B/8B-Base on DeepMath-103K) and competitive coding (DeepSeek-R1-Distill-Qwen-1.5B on ArcherCodeR):

- Consistently improves final accuracy over tuned weight-space AdamW / Muon baselines.
- Reaches matched accuracy with substantially fewer actor updates: **2.7× fewer** on Qwen3-8B-Base (ISO-AdamW reaches AdamW's 270-update score by update 100 and improves it from 0.495 to 0.509 by update 210), **2.2×** on Qwen3-4B-Base, **1.4×** for ISO-Muon vs. Muon.
- Adds only ~7% end-to-end RL step time (FP64 SVD polar retraction), off the rollout-dominated critical path.

**ISO-Merger** — data-free composition of shared-base RL experts:

*Recovery of each specialist's score in the single merged model:*

| Shared base | Specialist domain | Recovery of specialist score |
|---|---|---:|
| Qwen2.5-7B-Instruct (3 experts) | Coding | **106.2%** |
| | Tool Use | 99.8% |
| | Memory | 99.5% |
| DeepSeek-R1-Distill-Qwen-1.5B (2 experts) | Coding | **101.4%** |
| | Math | **101.7%** |

Without any post-merge data, rollouts, gradient updates, or distillation, the merged model recovers — and in several domains exceeds — each specialist's own score on its domain.

*Aggregate comparison against data-free merging baselines (average over all benchmarks):*

| Setting | Experts | Best baseline | ISO-Merger |
|---|---|---:|---:|
| Qwen2.5-7B-Instruct | coding + tool use + memory (3 experts) | 62.88 | **63.80** |
| DeepSeek-R1-Distill-Qwen-1.5B | coding + math (2 experts) | 43.52 | **44.38** |

Baselines include Task Arithmetic, TIES, TSV, RAM, and OrthoMerge-G-TIES. ISO-Merger also improves worst@4 by 1.62 / 1.36 points, indicating more consistent capability recovery across stochastic generations.

## 🚀 ISO-Merger: Quick Start

[`iso_merger.py`](iso_merger.py) is a self-contained script that merges K RL experts fine-tuned from the same base checkpoint into a single fixed-spectrum model. It requires only the checkpoints (HuggingFace format with safetensors shards) — no data, rollouts, or training.

### Requirements

```bash
pip install torch numpy safetensors
```

### Usage

```bash
python iso_merger.py \
    --base    /path/to/shared_base_model \
    --experts /path/to/expert_1 /path/to/expert_2 [/path/to/expert_3 ...] \
    --out     /path/to/merged_model \
    --device  cuda \
    --out-dtype bfloat16
```

The output directory is a self-contained HF checkpoint (tokenizer/config copied from the base) plus `retention_coefficients.json` with the per-tensor merge coefficients for reproducibility.

### What it does (per 2D weight matrix, in float64)

1. Thin SVD of the base `W₀ = U₀ Σ₀ V₀ᵀ` and of each expert; sign-canonicalize each expert's singular pairs against the base.
2. Project each expert's frame displacement onto the Stiefel tangent spaces at `(U₀, V₀)`.
3. Mask trailing singular modes (keep ratio `ρ_keep = 0.9`).
4. Solve a ridge-stabilized Gram system for retention coefficients targeting unit self-retention of every expert's first-order weight effect, then clip to `[0, 1.5]`.
5. Aggregate, re-project onto the tangent space, polar-retract, and reconstruct `W★ = U★ Σ₀ V★ᵀ` — the merged model shares the base spectrum up to floating-point error.

1D parameters (norm scales, biases) use a uniform task-vector average; all other tensors are copied from the base. All hyperparameters are frozen to the paper's configuration (Appendix E), so running with only `--base/--experts/--out` reproduces the paper's ISO-Merger models.

## 🏋️ ISO-Optimizer

ISO-Optimizer applies a chosen base optimizer (AdamW or Muon) to the frame variables `(U, V)` under the fixed base spectrum `Σ₀`, with an FP64 SVD-based polar retraction after each factor update. Our RLVR training runs are built on [verl](https://github.com/volcengine/verl) with DAPO.

**Code release coming soon.**

## 📋 TODO

- [x] Release ISO-Merger
- [ ] Release ISO-Optimizer (verl-based training code)

## 📖 Citation

If you find ISO useful, please consider citing:

```bibtex
@article{zhu2026iso,
  title   = {ISO: An RLVR-Native Optimization Stack},
  author  = {Zhu, Hanqing and Cong, Wenyan and Sha, Zhizhou and Mukherjee, Sagnik and Song, Xinyuan and Gonz{\'a}lez-Mart{\'i}nez, David and Wu, Xiaoxia and Tian, Yuandong and Liu, Shiwei and Pan, David Z. and Wang, Zhangyang},
  journal = {arXiv preprint arXiv:2607.19331},
  year    = {2026}
}
```

This work builds on our prior analysis of RLVR optimization dynamics:

```bibtex
@article{zhu2025path,
  title   = {The Path Not Taken: RLVR Provably Learns Off the Principals},
  author  = {Zhu, Hanqing and Zhang, Zhenyu and Huang, Hanxian and Su, DiJia and Liu, Zechun and Zhao, Jiawei and Fedorov, Igor and Pirsiavash, Hamed and Sha, Zhizhou and Lee, Jinwon and others},
  journal = {arXiv preprint arXiv:2511.08567},
  year    = {2025}
}
```

## 📬 Contact

- Hanqing Zhu: hqzhu@utexas.edu
- Wenyan Cong: wycong@utexas.edu
