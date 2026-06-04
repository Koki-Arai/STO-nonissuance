# Replication Package

## Security Token Offerings and Non-Issuance of Beneficiary Certificates: A Signalling Equilibrium Approach under Japanese Trust Law

**Author:** Koki Arai  
**Affiliation:** Faculty of Business Studies, Kyoritsu Women's University, Tokyo, Japan  
**Funding:** JSPS KAKENHI Grant Number 23K01404  
**Target Journal:** Financial Innovation (Springer)

---

## Overview

This repository provides the complete replication package for the numerical simulations and structural estimation reported in the paper. All code runs on **Google Colab** (free tier) without local installation.

The paper develops a signalling-equilibrium model for security token offerings (STOs) structured under Article 185(2) of the Japanese Trust Act, where junior beneficiary certificates are withheld from issuance. Five scripts are provided:

| # | Script | Purpose | Runtime (Colab) |
|---|--------|---------|----------------|
| 01 | `01_simulations.py` | Numerical verification of Propositions 1–6; Figures 1–7 | ~3 min |
| 02 | `02_smm_estimation.py` | Baseline SMM structural estimation (8 moments; Appendix B) | ~45–60 min |
| 03 | `03_smm7_robustness.py` | Robustness: re-estimation excluding Std[P_t/K] (7 moments; Online Appendix OA.2, Table OA.3) | ~35–45 min |
| 04 | `04_partial_recovery.py` | Dynamic extension under partial recovery (Online Appendix OA.6) | ~5 min |
| 05 | `05_sensitivity_analysis.py` | Welfare sensitivity to δσ perturbation (Online Appendix OA.3, Tables OA.4–OA.6) | ~3 min |

---

## Repository Structure

```
sto-nonissuance/
│
├── README.md
├── REPLICATION_GUIDE.md        ← Step-by-step instructions
├── requirements.txt
├── LICENSE                     ← MIT
├── .gitignore
│
├── notebooks/
│   ├── 01_simulations.py       ← Figures 1–7 / Propositions 1–6
│   ├── 02_smm_estimation.py    ← SMM estimation (Appendix B)
│   ├── 03_smm7_robustness.py   ← 7-moment robustness (OA.2)
│   ├── 04_partial_recovery.py  ← Partial recovery extension (OA.6)
│   └── 05_sensitivity_analysis.py  ← Welfare sensitivity (OA.3)
│
├── data/
│   └── smm_target_moments.csv  ← 8 SMM target moment conditions
│
└── figures/                    ← Output directory (created on first run)
```

---

## Quick Start on Google Colab

**For all five scripts, the procedure is the same:**

1. Open [colab.research.google.com](https://colab.research.google.com/)
2. Create a new notebook
3. Upload the script using:
   ```python
   from google.colab import files
   files.upload()   # select the .py file
   ```
4. Run all cells:
   ```python
   exec(open('filename.py').read())
   ```
5. Download output figures from the Files panel (left sidebar).

---

## SMM Estimation Results

### Baseline (8-moment; Appendix B, Table OA.1)

| Parameter | Estimate | SE | 95% CI |
|-----------|----------|----|--------|
| R(H) | 1.548 | 0.162 | [1.312, 1.999] |
| R(L) | 1.140 | 0.107 | [1.005, 1.400] |
| c_A | 0.219 | 0.032 | [0.163, 0.270] |
| c_b | 0.181 | 0.027 | [0.102, 0.200] |
| δ | 0.846 | 0.070 | [0.729, 0.982] † |
| σ | 0.851 | 0.074 | [0.733, 0.990] † |
| **δσ** | **0.720** | — | identified (§B.3) |
| λ_u | 0.081 | 0.010 | [0.056, 0.101] |
| s_θ | 0.229 | 0.044 | [0.138, 0.300] |
| Q | 0.000516 | — | — |

† δ and σ are not separately identified; only δσ = 0.720 is identified. The values δ = 0.846, σ = 0.851 are one decomposition consistent with the model. See Appendix B.3.

### 7-Moment Robustness (Excluding Std[P_t/K]; Online Appendix OA.2, Table OA.3)

| Parameter | 8-mom | 7-mom | Δ |
|-----------|-------|-------|---|
| R(H) | 1.548 | 1.549 | +0.001 |
| R(L) | 1.140 | 1.138 | −0.002 |
| c_A | 0.219 | 0.234 | +0.015 |
| c_b | 0.181 | 0.196 | +0.015 (at bound) |
| **δσ** | **0.720** | **0.720** | **−0.0003** ✓ |
| λ_u | 0.081 | 0.081 | 0.000 ✓ |
| Q | 0.000516 | 0.000478 | improved |
| Out-of-sample m₅ | — | −31.9% | structural misfit confirmed |

The welfare-relevant composite δσ and the upgrade-outage rate λ_u are unchanged when m₅ is excluded. The welfare conclusions of Propositions 3–5 are robust to this robustness check.

---

## Identification Note (Appendix B.3)

The parameters δ and σ are **not separately identified** from the eight moment conditions: all model implications depend on the product δσ = 0.720. The Jacobian ∂m(θ̂)/∂θ has two zero singular values confirming this. Individual values (δ = 0.846, σ = 0.851) represent one decomposition and should be interpreted as reference values, not separately estimated structural parameters.

Well-identified parameters: δσ = 0.720, R(H), R(L), λ_u, s_θ.

---

## Key Model Results

| Quantity | Value | Source |
|---------|-------|--------|
| Separation region | [0.646, 0.877) | Propositions 1–2, SMM |
| δσ (baseline) | 0.720 | SMM estimate |
| δσ_dom (dominance threshold) | 0.963 | K/(K+c_A−c_b) |
| Required improvement in δσ | 34 pp | Section 6.2 |
| W_A (conventional issuance) | 0.125 | SMM + Proposition 3 |
| W_B (opt-in STO) | −0.051 | SMM + Proposition 5 |
| W_C (mandatory STO) | −0.215 | SMM + Proposition 5 |
| ΔW(B−A) at baseline | −0.176 | W_B > W_C; W_A > W_B |
| Amplification ∂²P/∂λ_u∂δ | −3.20 | Dynamic Corollary |

---

## Computational Environment

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |

All packages are pre-installed in Google Colab. For local installation: `pip install -r requirements.txt`.

---

## Reproducibility

- All scripts use `RANDOM_SEED = 42`. Set `RANDOM_SEED = None` for fresh draws.
- The SMM Stage 1 result (Q = 0.000516) may vary slightly across runs due to Monte Carlo noise. Stage 2 is rejected when it degrades the criterion; Stage 1 estimates are always used for inference.
- Bootstrap valid-run counts (42–49/60 for 8-moment; 28/60 for 7-moment) are normal; invalid runs arise when Nelder-Mead fails parameter bounds.

---

## Citation

```bibtex
@article{arai2026sto,
  author  = {Arai, Koki},
  title   = {Security Token Offerings and Non-Issuance of Beneficiary
             Certificates: A Signalling Equilibrium Approach under
             Japanese Trust Law},
  journal = {Financial Innovation},
  year    = {2026},
  note    = {Forthcoming. JSPS KAKENHI 23K01404.}
}
```

---

## License

MIT License. See `LICENSE` for details.

## Contact

Koki Arai · Faculty of Business Studies, Kyoritsu Women's University, Tokyo · koki.arai@nifty.ne.jp · ORCID: 0000-0002-6907-4046
