# Replication Guide

Step-by-step instructions to reproduce all numerical results in the paper.

---

## Script 01 — Numerical Simulations (Figures 1–7)

**File:** `notebooks/01_simulations.py`  
**Paper sections:** Sections 3–6  
**Runtime:** ~3 min on Colab

### Cell structure

| Cell | Output | Paper location |
|------|--------|---------------|
| 1 | Imports | — |
| 2 | CONFIG (all parameters) | — |
| 3 | Core model functions | — |
| 4 | **Figure 1**: Separation region & IC | Section 4.1 |
| 5 | **Figure 2**: Equilibrium D* and welfare | Section 4.2 |
| 6 | **Figure 3**: Submodularity W*(Λ,Σ) | Section 5.1 |
| 7 | **Figure 4**: Three-regime welfare comparison | Section 5.2 |
| 8 | **Figure 5**: Dynamic token pricing | Section 5.3 |
| 9 | **Figure 6**: Adoption S-curve | Section 5.3 |
| 10 | **Figure 7**: Policy tier analysis | Section 6.2 |
| 11 | Numerical verification summary | Appendix B.4 |

### Expected Cell 11 output (key lines)

```
δσ̅  = K/R(H) = 0.6459       (separation region lower bound)
δσ̅̅  = K/R(L) = 0.8772       (upper bound)
Baseline δσ = 0.7200          in region: True
D*           = 1.3889
IC LHS = 0.0698  ≥  RHS = 0.0383:  True

∂²W*/∂Λ∂Σ at (1.5,1.5) = -0.0111  (< 0: True)

W_A = 0.1251   W_B = -0.0502   (W_A > W_B at baseline)
Dominance threshold δσ_dom = 0.9630  (outside separation region)

∂P/∂λ_u   = ... < 0      (True)
∂²P/∂λ_u∂δ = -3.20 < 0   (amplification confirmed)
```

---

## Script 02 — SMM Structural Estimation (Appendix B)

**File:** `notebooks/02_smm_estimation.py`  
**Paper sections:** Appendix B; Online Appendix OA.2  
**Runtime:** ~45–60 min on Colab

### Expected progress

**Stage 1 (Differential Evolution, ~20 min):**
```
differential_evolution step 1: f(x)= 0.191...
...
differential_evolution step ~254: f(x)= 0.000516
Stage-1 Q = 0.000516
```

**Stage 2 (Normalised weighting, ~5 min):**
```
Stage-2 rejected (Q >> Stage-1 Q). Using Stage-1.
```
Stage 2 rejection is expected and correct (see Appendix B.2).

**Bootstrap (60 replications, ~20 min):**
```
60/60  (valid: ~42–49)
```

**Final table:**
```
Param    Est      SE     95% CI
R(H)     1.548   0.162   [1.312, 1.999]
R(L)     1.140   0.107   [1.005, 1.400]
c_A      0.219   0.032   [0.163, 0.270]
c_b      0.181   0.027   [0.102, 0.200]
δ        0.846   0.070   [0.729, 0.982] †
σ        0.851   0.074   [0.733, 0.990] †
δσ       0.720   (identified)
λ_u      0.081   0.010   [0.056, 0.101]
s_θ      0.229   0.044   [0.138, 0.300]
Q = 0.000516
```

---

## Script 03 — 7-Moment Robustness (Online Appendix OA.2, Table OA.3)

**File:** `notebooks/03_smm7_robustness.py`  
**Purpose:** Re-estimate excluding Std[P_t/K] (m₅) to test robustness  
**Runtime:** ~35–45 min on Colab

### Expected key output

```
ROBUSTNESS VERDICT:
  δσ: 0.7199 (8-mom) → 0.7196 (7-mom)   Δ = -0.0003  ✓
  λ_u: 0.081 (8-mom) → 0.081 (7-mom)    Δ = 0.000    ✓
  Q (7-mom) = 0.000478  (improved from 0.000516)

  Out-of-sample m₅ at 7-mom estimates: -31.9%
  (same order as baseline -31.6% → structural, not artefact)

  c_b: 0.181 → 0.196  ← BOUND (upper bound 0.200)
```

**Interpretation:** Welfare-relevant parameters are unchanged. The price-volatility misfit is structural to the OU specification, not an artefact of m₅'s inclusion.

---

## Script 04 — Partial Recovery Extension (Online Appendix OA.6)

**File:** `notebooks/04_partial_recovery.py`  
**Purpose:** Dynamic model with partial cash-flow recovery after failures (ϕ ∈ [0,1])  
**Runtime:** ~5 min on Colab

### Expected key output

```
ϕ = 0.00:  ∂²P/∂λ_u∂δ = −3.200  (baseline, conservative)
ϕ = 0.25:  ∂²P/∂λ_u∂δ = −2.246
ϕ = 0.50:  ∂²P/∂λ_u∂δ = −1.391
ϕ = 0.75:  ∂²P/∂λ_u∂δ = −0.633
ϕ = 1.00:  ∂²P/∂λ_u∂δ = 0 (full recovery = no amplification)

Welfare ranking (Props 3–5): INVARIANT to ϕ ∈ [0,1)
Baseline ϕ = 0 is the conservative case.
```

---

## Script 05 — Welfare Sensitivity Analysis (Online Appendix OA.3)

**File:** `notebooks/05_sensitivity_analysis.py`  
**Purpose:** Welfare rankings under δσ perturbation (±0.05, ±0.10)  
**Runtime:** ~3 min on Colab

### Expected Table OA.4 (key rows)

```
δσ    W_A     W_B      W_C      ΔW(B−A)   Ranking
0.700  0.125  -0.070  -0.235   -0.195    A > B > C
0.720  0.125  -0.051  -0.215   -0.176    A > B > C  ← baseline
0.750  0.125  -0.023  -0.187   -0.148    A > B > C
0.800  0.125  +0.019  -0.146   -0.106    A > B > C
0.850  0.125  +0.056  -0.109   -0.069    A > B > C
```

W_B > W_C throughout; W_A > W_B throughout. Full reversal requires δσ ≥ 0.963.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Stage 2 Q >> Stage 1 Q | Ill-conditioned W (expected) | Stage 1 used automatically |
| R(H) > 2.0 after optimisation | polish=True (not default in v3) | Ensure polish=False |
| c_b at upper bound 0.200 | True optimum near boundary | Results still valid; reported in text |
| Bootstrap valid < 30/60 | Tight bounds | Extend relevant bounds slightly |
| Colab session timeout | >12h runtime | Reduce N_BOOT; run stages separately |

---

## Correspondence: Code ↔ Paper

| Paper location | Code location |
|---------------|--------------|
| Eq. (3): D* = K/(δσ) | `Dstar()` in 01, Cell 3 |
| Eq. (6): W*(δσ) | `W_B()` in 01, Cell 3 |
| Eq. (10): ∂²W*/∂Λ∂Σ < 0 | Numerical cross-partial, 01, Cell 6 |
| Eq. (14): P_t = δσG(θ_t) | `sim_price()` in 01, Cell 3 |
| Appendix B.2: SMM estimator | Cells 4–6 in 02 |
| Appendix B.3: Identification | Cell 8 in 02 |
| Table OA.1: Baseline estimates | Cell 9 in 02 |
| Table OA.3: 7-moment robustness | Cells 7–9 in 03 |
| Table OA.6: Partial recovery amplification | Cell 5 in 04 |
| Table OA.4: δσ sensitivity | Cell 4 in 05 |
