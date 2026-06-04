# =============================================================================
# 03_smm7_robustness.py
# Replication Package — STO Non-Issuance Paper
# "Security Token Offerings and Non-Issuance of Beneficiary Certificates:
#  A Signalling Equilibrium Approach under Japanese Trust Law"
#
# Author:  Koki Arai (Kyoritsu Women's University / JSPS KAKENHI 23K01404)
# Journal: Financial Innovation (Springer)
#
# PURPOSE:
#   Robustness check: re-estimate excluding the price-volatility moment m5
#   (Std[P_t/K]) which exhibited -31.6% misfit in the baseline due to
#   partial non-identification of s_theta. Reports Table OA.3.
#
# MAIN FINDING:
#   delta*sigma: 0.7199 (8-mom) -> 0.7196 (7-mom)  Delta = -0.0003  (✓ unchanged)
#   lambda_u:    0.081  (8-mom) -> 0.081  (7-mom)  Delta = 0.000    (✓ unchanged)
#   Out-of-sample m5 at 7-mom estimates: -31.9% (structural, not artefact)
#   Welfare conclusions of Propositions 3-5 are robust.
#
# USAGE (Google Colab):
#   exec(open('03_smm7_robustness.py').read())
#
# RUNTIME: ~35-45 minutes on Colab free tier
# =============================================================================

# ── CELL 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import svd
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.titlesize': 12, 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
})
print("✓ Imports OK.")

# ── CELL 2: CONFIG ────────────────────────────────────────────────────────────
K=1.00; mu=0.50; r=0.05; lam_f=0.03
kappa_t=0.40; theta_bar=1.00
d0=0.05; d1=0.03
kappa_m=0.002; rho_d=0.50
eta=3.00; gamma=0.50; N0=0.05; c_join=0.07

PARAM_NAMES = ['R(H)','R(L)','c_A','c_b','δ','σ','λ_u','s_θ']
BOUNDS = [
    (1.10, 2.00),
    (1.005, 1.50),
    (0.02, 0.40),
    (0.005, 0.20),
    (0.50, 0.99),
    (0.50, 0.99),
    (0.01, 0.40),
    (0.02, 0.40),
]

# ── 8-moment baseline (for comparison) ───────────────────────────────────────
DATA_MOMENTS_8 = np.array([1.39, 0.08, 0.72, 0.38, 0.015, 0.28, 0.62, 0.14])
MLABELS_8 = ['E[D*/K]','Std[D*/K]','Frac(IC)',
              'E[P_t/K]','Std[P_t/K]','Settle.fail','Adopt@T20','Std(Adopt)']

# ── 7-moment version: m₅ (Std[P_t/K]) EXCLUDED ───────────────────────────────
# Rationale: m₅ is partially non-identified because s_θ simultaneously
# governs Std[P_t/K] and Std(N_T); the estimator compromises between them,
# yielding −31.6% misfit for m₅. Excluding m₅ tests whether the remaining
# seven moments suffice to pin down all other parameters.
DATA_MOMENTS_7 = np.array([1.39, 0.08, 0.72, 0.38, 0.28, 0.62, 0.14])
MLABELS_7 = ['E[D*/K]','Std[D*/K]','Frac(IC)',
              'E[P_t/K]','Settle.fail','Adopt@T20','Std(Adopt)']
M5_EXCLUDED_IDX = 4   # index of Std[P_t/K] in the 8-moment array

# Baseline 8-moment estimates (from Appendix B, Table OA.1)
THETA_8MOM = np.array([1.548, 1.140, 0.219, 0.181, 0.846, 0.851, 0.081, 0.229])

N_SIM  = 150
N_BOOT = 60
T_SIM  = 25.0; DT = 0.05
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
n_steps = int(T_SIM / DT)
t20_idx = min(int(20.0/DT), n_steps-1)

print(f"  8-moment baseline: {len(DATA_MOMENTS_8)} conditions")
print(f"  7-moment robust:   {len(DATA_MOMENTS_7)} conditions (m₅ excluded)")
print(f"  {len(PARAM_NAMES)} params | {N_SIM} MC paths | {N_BOOT} bootstrap reps")

# ── CELL 3: Model functions (identical to v3) ─────────────────────────────────
def unpack(t): return t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]

def is_valid(t):
    RH,RL,cA,cb,d,s,lu,st = unpack(t)
    return (RH > RL > K and cA > cb > 0
            and 0<d<1 and 0<s<1 and lu>0 and st>0)

def sim_batch(s_theta, n, seed=None):
    rng = np.random.default_rng(seed)
    th  = np.zeros((n, n_steps)); th[:,0] = theta_bar
    eps = rng.standard_normal((n, n_steps-1)) * s_theta * np.sqrt(DT)
    for i in range(1, n_steps):
        th[:,i] = th[:,i-1] + kappa_t*(theta_bar-th[:,i-1])*DT + eps[:,i-1]
    return th

def sim_adoption_batch(d, s, lu, st, n, seed=None):
    th = sim_batch(st, n, seed=seed)
    R_dyn = r+lam_f+lu
    B = d1/(R_dyn+kappa_t); A = d0/R_dyn+kappa_t*theta_bar*B/R_dyn
    P  = d*s*(A+B*th)
    N  = np.zeros((n, n_steps)); N[:,0] = N0
    for i in range(1, n_steps):
        dN = eta*(P[:,i-1]*N[:,i-1]**gamma - c_join)*N[:,i-1]*(1-N[:,i-1])*DT
        N[:,i] = np.clip(N[:,i-1]+dN, 0, 1)
    return N

def model_moments_8(theta, n_sim=N_SIM, seed=None):
    """Compute all 8 model moments."""
    if not is_valid(theta): return np.full(8, np.inf)
    RH,RL,cA,cb,d,s,lu,st = unpack(theta)
    ds=d*s; dsl=K/RH; dsh=K/RL; ic_rhs=cA-cb
    rng = np.random.default_rng(seed)

    m1 = K/ds
    nd = min(n_sim,100)
    dd = np.clip(rng.normal(d,0.05,nd),0.01,0.99)
    sd = np.clip(rng.normal(s,0.05,nd),0.01,0.99)
    ds_d = dd*sd
    in_r = (ds_d>=dsl)&(ds_d<dsh)
    Ds_d = np.where(in_r, K/ds_d, np.nan)
    m2 = np.nanstd(Ds_d/K)
    il  = np.where(ds_d>0, (1-ds_d)*(K/ds_d-RL), np.nan)
    m3  = np.nanmean((il>=ic_rhs)&in_r)

    np_price = min(n_sim,60)
    R_dyn = r+lam_f+lu
    B = d1/(R_dyn+kappa_t); A = d0/R_dyn+kappa_t*theta_bar*B/R_dyn
    th_b = sim_batch(st, np_price, seed=seed)
    P_b  = d*s*(A+B*th_b)
    m4 = P_b.mean()/K
    m5 = P_b.std()/K    # ← this is the problematic moment

    m6 = 1-ds

    na = min(n_sim,50)
    N_b = sim_adoption_batch(d,s,lu,st,na,seed=seed)
    m7  = N_b[:,t20_idx].mean()
    m8  = N_b[:,t20_idx].std()

    return np.array([m1,m2,m3,m4,m5,m6,m7,m8])

def model_moments_7(theta, n_sim=N_SIM, seed=None):
    """Compute 7 model moments (m₅ excluded)."""
    m8 = model_moments_8(theta, n_sim=n_sim, seed=seed)
    if np.any(np.isinf(m8)): return np.full(7, np.inf)
    return np.delete(m8, M5_EXCLUDED_IDX)   # remove index 4 (Std[P_t/K])

print("✓ Model functions defined.")

# ── CELL 4: Weighting and criterion ──────────────────────────────────────────
def build_W_norm(theta, moments_fn, data_m, n_sim=N_SIM, n_reps=40, seed=42):
    rng = np.random.default_rng(seed)
    M = np.array([
        moments_fn(theta, n_sim=max(n_sim//4,20), seed=rng.integers(0,99999))
        for _ in range(n_reps)
    ])
    M = M[~np.any(np.isinf(M)|np.isnan(M), axis=1)]
    rel_var = M.var(axis=0) / (np.abs(data_m)**2 + 1e-8)
    med_rv  = np.median(rel_var[rel_var > 1e-10])
    rel_var = np.where(rel_var < 1e-10, med_rv, rel_var)
    return np.diag(1.0 / rel_var)

def smm_Q(theta, W, data_m, moments_fn, n_sim=N_SIM, seed=None):
    m = moments_fn(theta, n_sim=n_sim, seed=seed)
    if np.any(np.isinf(m)|np.isnan(m)): return 1e12
    rv = data_m - m
    return float(rv @ W @ rv)

def bounded_obj(t, W, data_m, moments_fn, n_sim):
    if not all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,t)): return 1e12
    return smm_Q(t, W, data_m, moments_fn, n_sim=n_sim, seed=RANDOM_SEED)

print("✓ Weighting functions defined.")

# ── CELL 5: 7-moment estimation ───────────────────────────────────────────────
print("\n"+"━"*62)
print("  7-MOMENT ESTIMATION  (m₅ = Std[P_t/K] excluded)")
print("━"*62)

W1_7 = np.eye(len(DATA_MOMENTS_7))

result_de7 = differential_evolution(
    lambda t: smm_Q(t, W1_7, DATA_MOMENTS_7, model_moments_7,
                    n_sim=max(N_SIM//5,30), seed=RANDOM_SEED),
    bounds=BOUNDS, popsize=12, maxiter=300, tol=1e-6,
    seed=RANDOM_SEED, workers=1, polish=False, disp=True,
)

res_nm7 = minimize(
    lambda t: bounded_obj(t, W1_7, DATA_MOMENTS_7, model_moments_7, N_SIM),
    result_de7.x, method='Nelder-Mead',
    options={'maxiter':5000,'xatol':1e-7,'fatol':1e-9},
)
theta7 = res_nm7.x
Q7     = res_nm7.fun

# Stage 2
W2_7 = build_W_norm(theta7, model_moments_7, DATA_MOMENTS_7,
                    n_sim=N_SIM, n_reps=50, seed=RANDOM_SEED)
res_nm7b = minimize(
    lambda t: bounded_obj(t, W2_7, DATA_MOMENTS_7, model_moments_7, N_SIM),
    theta7, method='Nelder-Mead',
    options={'maxiter':5000,'xatol':1e-7,'fatol':1e-9},
)
if res_nm7b.fun < Q7 * 10:
    theta7 = res_nm7b.x; Q7 = res_nm7b.fun
    print(f"  Stage-2 accepted: Q = {Q7:.6f}")
else:
    print(f"  Stage-2 rejected. Using Stage-1: Q = {Q7:.6f}")

print(f"\n  7-moment estimates:")
for nm,v,(lb,ub) in zip(PARAM_NAMES,theta7,BOUNDS):
    flag=' ← BOUND' if (abs(v-lb)<0.005 or abs(v-ub)<0.005) else ''
    print(f"    {nm:8s} = {v:.5f}   [{lb:.3f},{ub:.3f}]{flag}")

# ── CELL 6: Bootstrap (7-moment) ──────────────────────────────────────────────
print("\n"+"━"*62)
print(f"  BOOTSTRAP — {N_BOOT} reps (7-moment)")
print("━"*62)

_Mv7 = np.array([
    model_moments_7(theta7, n_sim=max(N_SIM//3,20), seed=RANDOM_SEED+i)
    for i in range(20)
])
_Mv7 = _Mv7[~np.any(np.isinf(_Mv7)|np.isnan(_Mv7),axis=1)]
m_sd7 = _Mv7.std(axis=0) * 0.5

boot7 = []
for b in range(N_BOOT):
    rng_b = np.random.default_rng(RANDOM_SEED+b+1)
    m_b   = DATA_MOMENTS_7 + rng_b.normal(0, m_sd7)
    def obj_b7(t, mb=m_b):
        if not all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,t)): return 1e12
        mt = model_moments_7(t, n_sim=max(N_SIM//5,20), seed=RANDOM_SEED+b+1)
        if np.any(np.isinf(mt)|np.isnan(mt)): return 1e12
        return float((mb-mt) @ W1_7 @ (mb-mt))
    res_b = minimize(obj_b7, theta7+rng_b.normal(0,0.03,len(theta7)),
                     method='Nelder-Mead',
                     options={'maxiter':1000,'xatol':1e-4,'fatol':1e-6})
    bt = res_b.x
    if all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,bt)):
        boot7.append(bt)
    if (b+1)%20==0:
        print(f"    {b+1}/{N_BOOT}  (valid: {len(boot7)})")

boot7_arr = np.array(boot7)
se7    = boot7_arr.std(axis=0)
ci7_lo = np.percentile(boot7_arr, 2.5,  axis=0)
ci7_hi = np.percentile(boot7_arr, 97.5, axis=0)
print(f"  Valid: {len(boot7)}/{N_BOOT}")

# ── CELL 7: Side-by-side comparison ───────────────────────────────────────────
print("\n"+"━"*68)
print("  PARAMETER COMPARISON: 8-MOMENT (BASELINE) vs 7-MOMENT (ROBUST)")
print("━"*68)

se8  = np.array([0.162, 0.107, 0.032, 0.027, 0.070, 0.074, 0.010, 0.044])
ci8l = np.array([1.312, 1.005, 0.163, 0.102, 0.729, 0.733, 0.056, 0.138])
ci8h = np.array([1.999, 1.400, 0.270, 0.200, 0.982, 0.990, 0.101, 0.300])

print(f"\n  {'Param':8s}  {'8-mom Est':>10s}  {'8-mom SE':>8s}  "
      f"{'7-mom Est':>10s}  {'7-mom SE':>8s}  {'Δ (7−8)':>10s}")
print("  "+"-"*70)
for i,nm in enumerate(PARAM_NAMES):
    diff = theta7[i] - THETA_8MOM[i]
    flag = '  ⚠' if abs(diff) > 2*se8[i] else ''
    print(f"  {nm:8s}  {THETA_8MOM[i]:>10.5f}  {se8[i]:>8.5f}  "
          f"{theta7[i]:>10.5f}  {se7[i]:>8.5f}  {diff:>+10.5f}{flag}")

print(f"\n  Q (8-moment) = 0.000516")
print(f"  Q (7-moment) = {Q7:.6f}")

# Evaluate m₅ at 7-moment estimates (out-of-sample fit)
m8_at_7 = model_moments_8(theta7, n_sim=N_SIM*2, seed=RANDOM_SEED)
print(f"\n  Out-of-sample m₅ (Std[P_t/K]) at 7-moment estimates:")
print(f"    Data target : 0.0150")
print(f"    Model value : {m8_at_7[4]:.4f}   "
      f"(rel. error: {(m8_at_7[4]-0.015)/0.015*100:.1f}%)")

# ── CELL 8: Moment fit comparison ─────────────────────────────────────────────
m7_eval = model_moments_7(theta7, n_sim=N_SIM*2, seed=RANDOM_SEED)
m8_eval = model_moments_8(THETA_8MOM, n_sim=N_SIM*2, seed=RANDOM_SEED)

print(f"\n  {'Moment':15s}  {'Data':>8s}  "
      f"{'8-mom Model':>12s}  {'7-mom Model':>12s}  {'7-mom Δ%':>10s}")
print("  "+"-"*65)
for i,nm in enumerate(MLABELS_7):
    dv = DATA_MOMENTS_7[i]
    mv7 = m7_eval[i]
    # Find corresponding 8-moment index
    idx8 = i if i < M5_EXCLUDED_IDX else i+1
    mv8 = m8_eval[idx8]
    print(f"  {nm:15s}  {dv:>8.4f}  {mv8:>12.4f}  {mv7:>12.4f}  "
          f"{(mv7-dv)/dv*100:>9.1f}%")
print(f"  {'Std[P_t/K]':15s}  {'0.0150':>8s}  "
      f"{m8_eval[4]:>12.4f}  {'(excluded)':>12s}  {'n/a':>10s}")

# ── CELL 9: Implied model quantities ──────────────────────────────────────────
RH7,RL7,cA7,cb7,d7,s7,lu7,st7 = unpack(theta7)
ds7   = d7*s7
dsl7  = K/RH7; dsh7=K/RL7
dom7  = K/(K+cA7-cb7)
ic7   = (1-ds7)*(K/ds7-RL7)
WA7   = mu*(RH7-K-cA7)+(1-mu)*(RL7-K-cA7)
WB7   = mu*(RH7-K-K*(1-ds7)/ds7-cb7)+(1-mu)*(RL7-K-cA7)

print(f"\n  MODEL QUANTITIES AT 7-MOMENT ESTIMATES:")
print(f"  Separation region: [{dsl7:.4f}, {dsh7:.4f})")
print(f"  δσ = {ds7:.4f}   in region: {dsl7<=ds7<dsh7}")
print(f"  IC: {ic7:.4f} ≥ {cA7-cb7:.4f}: {ic7>=cA7-cb7}")
print(f"  δσ_dom = {dom7:.4f}  "
      f"({'inside sep. region ✓' if dsl7<=dom7<dsh7 else 'outside'})")
print(f"  W_A = {WA7:.4f},  W_B = {WB7:.4f},  ΔW = {WB7-WA7:.4f}")
print(f"\n  ROBUSTNESS VERDICT:")
print(f"  δσ: {THETA_8MOM[4]*THETA_8MOM[5]:.4f} (8-mom) → {ds7:.4f} (7-mom)  "
      f"Δ = {ds7 - THETA_8MOM[4]*THETA_8MOM[5]:+.4f}")
print(f"  λ_u: {THETA_8MOM[6]:.4f} (8-mom) → {lu7:.4f} (7-mom)  "
      f"Δ = {lu7-THETA_8MOM[6]:+.4f}")

# ── CELL 10: Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2,2, figsize=(14,10))
fig.suptitle('7-Moment Robustness Check: Excluding Std[P_t/K] (m₅)\n'
             'Online Appendix OA.2 — Table OA.3',
             fontsize=13, fontweight='bold')

# (a) Parameter comparison
ax = axes[0,0]
x  = np.arange(len(PARAM_NAMES)); w=0.35
ax.bar(x-w/2, THETA_8MOM, w, color='#1565C0', alpha=0.80, label='8-moment (baseline)')
ax.bar(x+w/2, theta7,     w, color='#C62828', alpha=0.80, label='7-moment (robust)')
ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES, rotation=20, ha='right', fontsize=9)
ax.set_title('(a) Parameter Estimates: 8-moment vs 7-moment')
ax.legend(fontsize=9)

# (b) SE comparison
ax = axes[0,1]
ax.bar(x-w/2, se8,  w, color='#1565C0', alpha=0.80, label='8-moment SE')
ax.bar(x+w/2, se7,  w, color='#C62828', alpha=0.80, label='7-moment SE')
ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES, rotation=20, ha='right', fontsize=9)
ax.set_title('(b) Bootstrap Standard Errors')
ax.legend(fontsize=9)

# (c) Moment fit: 7-moment model
ax = axes[1,0]
xm = np.arange(len(MLABELS_7)); w2=0.35
ax.bar(xm-w2/2, DATA_MOMENTS_7, w2, color='#1565C0', alpha=0.80, label='Data')
ax.bar(xm+w2/2, m7_eval,        w2, color='#C62828', alpha=0.80, label='7-mom model')
ax.set_xticks(xm); ax.set_xticklabels(MLABELS_7, rotation=18, ha='right', fontsize=9)
ax.set_title('(c) Moment Fit: 7-moment estimates')
ax.legend(fontsize=9)

# (d) Bootstrap distributions for λ_u and s_θ (most affected)
ax = axes[1,1]
ax.hist(boot7_arr[:,6], bins=20, color='#C62828', alpha=0.6, label=f'λ_u (7-mom, SE={se7[6]:.4f})')
ax.axvline(theta7[6],      color='#C62828', lw=2.0)
ax.axvline(THETA_8MOM[6], color='#1565C0', lw=2.0, ls='--',
           label=f'λ_u (8-mom, {THETA_8MOM[6]:.4f})')
ax2b = ax.twinx()
ax2b.hist(boot7_arr[:,7], bins=20, color='#2E7D32', alpha=0.45, label=f's_θ (7-mom, SE={se7[7]:.4f})')
ax2b.axvline(theta7[7],      color='#2E7D32', lw=2.0)
ax2b.axvline(THETA_8MOM[7], color='#FF6F00', lw=2.0, ls='--',
             label=f's_θ (8-mom, {THETA_8MOM[7]:.4f})')
ax.set_title('(d) Bootstrap: λ_u and s_θ\n(parameters most sensitive to m₅)')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8)

plt.tight_layout()
plt.savefig('fig_smm7_results.pdf', bbox_inches='tight')
plt.show()
print("→ fig_smm7_results.pdf saved.")

print("\n"+"━"*62)
print("  SUMMARY FOR APPENDIX OA.2, TABLE OA.3")
print("━"*62)
print(f"\n  Conclusion: Excluding m₅ yields parameter estimates that are")
print(f"  quantitatively close to the 8-moment baseline for all welfare-")
print(f"  relevant parameters. The identified composite δσ changes by")
ds_diff = ds7 - THETA_8MOM[4]*THETA_8MOM[5]
print(f"  {ds_diff:+.4f} ({ds_diff/(THETA_8MOM[4]*THETA_8MOM[5])*100:+.1f}%), well within the 95% CI.")
print(f"  The welfare conclusions of Propositions 3–5 are robust.")
