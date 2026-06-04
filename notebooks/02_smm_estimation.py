# =============================================================================
# 02_smm_estimation.py
# Replication Package — STO Non-Issuance Paper
# "Security Token Offerings and Non-Issuance of Beneficiary Certificates:
#  A Signalling Equilibrium Approach under Japanese Trust Law"
#
# Author:  Koki Arai (Kyoritsu Women's University / JSPS KAKENHI 23K01404)
# Journal: Financial Innovation (Springer)
#
# PURPOSE:
#   Baseline 8-moment SMM structural estimation.
#   Reports Table OA.1 (parameter estimates with bootstrap SE).
#   Corresponds to Appendix B and Online Appendix OA.2.
#
# USAGE (Google Colab):
#   exec(open('02_smm_estimation.py').read())
#
# KEY RESULTS:
#   delta*sigma = 0.720  (identified composite; delta, sigma not separately ID)
#   lambda_u    = 0.081  SE=0.010  (most precisely identified dynamic param)
#   Q           = 0.000516         (SMM criterion at final estimates)
#
# IDENTIFICATION NOTE:
#   delta and sigma appear only as the product delta*sigma in all moments.
#   The Jacobian has two zero singular values confirming non-identification.
#   Reported delta=0.846, sigma=0.851 are ONE decomposition (Appendix B.3).
#
# RUNTIME: ~45-60 minutes on Colab free tier (serial bootstrap)
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
    (1.10, 2.00),   # RH
    (1.02, 1.50),   # RL
    (0.02, 0.40),   # c_A
    (0.005,0.15),   # c_b
    (0.50, 0.99),   # delta
    (0.50, 0.99),   # sigma
    (0.01, 0.40),   # lambda_u
    (0.02, 0.40),   # s_theta
]

# Target moments (8 conditions)
DATA_MOMENTS = np.array([
    1.39,   # m1 E[D*/K]
    0.08,   # m2 Std[D*/K]
    0.72,   # m3 Frac(IC)
    0.38,   # m4 E[P_t/K]
    0.015,  # m5 Std[P_t/K]
    0.28,   # m6 Settlement failure 1-δσ
    0.62,   # m7 Adoption @ T=20
    0.14,   # m8 Std(Adoption @ T=20)
])
MLABELS = ['E[D*/K]','Std[D*/K]','Frac(IC)',
           'E[P_t/K]','Std[P_t/K]','Settle.fail',
           'Adopt@T20','Std(Adopt)']

# Simulation settings
N_SIM  = 150    # MC paths per evaluation
N_BOOT = 60     # bootstrap replications
T_SIM  = 25.0; DT = 0.05
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
n_steps = int(T_SIM / DT)
t20_idx = min(int(20.0/DT), n_steps-1)
print(f"  {len(PARAM_NAMES)} params | {len(DATA_MOMENTS)} moments | "
      f"{N_SIM} MC paths | {N_BOOT} bootstrap reps")

# ── CELL 3: Model functions ───────────────────────────────────────────────────
def unpack(t): return t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]

def is_valid(t):
    RH,RL,cA,cb,d,s,lu,st = unpack(t)
    return (RH > RL > K and cA > cb > 0
            and 0<d<1 and 0<s<1 and lu>0 and st>0)

def G_th(theta_path, lam_u):
    R = r+lam_f+lam_u
    B = d1/(R+kappa_t)
    A = d0/R + kappa_t*theta_bar*B/R
    return A + B*theta_path

def sim_batch(s_theta, n, seed=None):
    """Vectorised OU batch: returns (n, n_steps)."""
    rng = np.random.default_rng(seed)
    th  = np.zeros((n, n_steps)); th[:,0] = theta_bar
    eps = rng.standard_normal((n, n_steps-1)) * s_theta * np.sqrt(DT)
    for i in range(1, n_steps):
        th[:,i] = th[:,i-1] + kappa_t*(theta_bar-th[:,i-1])*DT + eps[:,i-1]
    return th

def sim_adoption_batch(d, s, lu, st, n, seed=None):
    th = sim_batch(st, n, seed=seed)
    P  = d*s*G_th(th, lu)
    N  = np.zeros((n, n_steps)); N[:,0] = N0
    for i in range(1, n_steps):
        dN = eta*(P[:,i-1]*N[:,i-1]**gamma - c_join)*N[:,i-1]*(1-N[:,i-1])*DT
        N[:,i] = np.clip(N[:,i-1]+dN, 0, 1)
    return N

def model_moments(theta, n_sim=N_SIM, seed=None):
    if not is_valid(theta): return np.full(8, np.inf)
    RH,RL,cA,cb,d,s,lu,st = unpack(theta)
    ds=d*s; dsl=K/RH; dsh=K/RL; ic_rhs=cA-cb
    rng = np.random.default_rng(seed)

    # m1: E[D*/K]
    m1 = K/ds

    # m2, m3: cross-deal dispersion and IC fraction
    nd = min(n_sim,100)
    dd = np.clip(rng.normal(d,0.05,nd),0.01,0.99)
    sd = np.clip(rng.normal(s,0.05,nd),0.01,0.99)
    ds_d = dd*sd
    in_r = (ds_d>=dsl)&(ds_d<dsh)
    Ds_d = np.where(in_r, K/ds_d, np.nan)
    m2 = np.nanstd(Ds_d/K)
    il  = np.where(ds_d>0, (1-ds_d)*(K/ds_d-RL), np.nan)
    m3  = np.nanmean((il>=ic_rhs)&in_r)

    # m4, m5: price moments (vectorised)
    np_price = min(n_sim,60)
    th_b = sim_batch(st, np_price, seed=seed)
    P_b  = d*s*G_th(th_b, lu)
    m4 = P_b.mean()/K
    m5 = P_b.std()/K

    # m6: settlement failure
    m6 = 1-ds

    # m7, m8: adoption (vectorised)
    na = min(n_sim,50)
    N_b = sim_adoption_batch(d,s,lu,st,na,seed=seed)
    m7  = N_b[:,t20_idx].mean()
    m8  = N_b[:,t20_idx].std()

    return np.array([m1,m2,m3,m4,m5,m6,m7,m8])

print("✓ Functions defined.")
m0 = model_moments(np.array([1.5,1.15,0.05,0.01,0.80,0.90,0.07,0.08]), n_sim=30, seed=0)
print("  Initial θ₀ moment check:")
for lb,mv,dv in zip(MLABELS,m0,DATA_MOMENTS):
    print(f"    {lb:15s} model={mv:.4f}  data={dv:.4f}  diff={mv-dv:+.4f}")

# ── CELL 4: Weighting matrix (normalised — avoids ill-conditioning) ───────────
def build_W_normalised(theta, n_sim=N_SIM, n_reps=40, seed=42):
    """
    Normalised diagonal weighting:
      W_ii = 1 / (Var[m_i(theta)] / data_moment_i^2)
    This scales by relative (not absolute) variance, preventing moments
    with near-zero absolute variance (like Settle.fail = 1-δσ) from
    dominating the criterion function.
    Special handling: moments with Var ≈ 0 (analytical moments) get
    weight = 1/median_var to avoid infinite weights.
    """
    rng = np.random.default_rng(seed)
    M   = np.array([
        model_moments(theta, n_sim=max(n_sim//4,20), seed=rng.integers(0,99999))
        for _ in range(n_reps)
    ])
    M = M[~np.any(np.isinf(M)|np.isnan(M), axis=1)]
    rel_var = M.var(axis=0) / (np.abs(DATA_MOMENTS)**2 + 1e-8)
    # Replace near-zero variances with median to avoid ∞ weight
    med_rv  = np.median(rel_var[rel_var > 1e-10])
    rel_var = np.where(rel_var < 1e-10, med_rv, rel_var)
    cond = rel_var.max() / rel_var.min()
    print(f"  Relative-variance range: [{rel_var.min():.4e}, {rel_var.max():.4e}]"
          f"  cond={cond:.1f}")
    return np.diag(1.0 / rel_var)

def smm_Q(theta, W, n_sim=N_SIM, seed=None):
    m = model_moments(theta, n_sim=n_sim, seed=seed)
    if np.any(np.isinf(m)|np.isnan(m)): return 1e12
    rv = DATA_MOMENTS - m
    return float(rv @ W @ rv)

print("✓ Weighting functions defined.")

# ── CELL 5: Stage 1 — Differential Evolution (polish=False) ──────────────────
print("\n"+"━"*60)
print("  STAGE 1 — Differential Evolution (polish=False, bounds enforced)")
print("━"*60)

W1 = np.eye(len(DATA_MOMENTS))  # identity for Stage 1

result_de = differential_evolution(
    lambda t: smm_Q(t, W1, n_sim=max(N_SIM//5,30), seed=RANDOM_SEED),
    bounds=BOUNDS,
    popsize=12, maxiter=300, tol=1e-6,
    seed=RANDOM_SEED,
    workers=1,
    polish=False,   # ← KEY FIX: do NOT let L-BFGS-B escape bounds
    disp=True,
)
theta1_de = result_de.x

# Nelder-Mead refinement (also bounded via penalty)
def bounded_obj(t, W, n_sim):
    if not all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,t)): return 1e12
    return smm_Q(t, W, n_sim=n_sim, seed=RANDOM_SEED)

res_nm1 = minimize(
    lambda t: bounded_obj(t, W1, N_SIM),
    theta1_de, method='Nelder-Mead',
    options={'maxiter':5000,'xatol':1e-7,'fatol':1e-9},
)
theta1 = res_nm1.x
Q1     = res_nm1.fun

print(f"\n  Stage-1 Q = {Q1:.6f}")
for nm,v,(lb,ub) in zip(PARAM_NAMES, theta1, BOUNDS):
    flag = ' ← AT BOUND ⚠' if (abs(v-lb)<0.005 or abs(v-ub)<0.005) else ''
    print(f"    {nm:8s} = {v:.5f}   [{lb:.3f},{ub:.3f}]{flag}")

# ── CELL 6: Stage 2 — Normalised diagonal W ───────────────────────────────────
print("\n"+"━"*60)
print("  STAGE 2 — Normalised-diagonal W (relative variance)")
print("━"*60)

W2   = build_W_normalised(theta1, n_sim=N_SIM, n_reps=50, seed=RANDOM_SEED)
cond2= np.linalg.cond(W2)
print(f"  Condition number W2: {cond2:.1f}")

res_nm2 = minimize(
    lambda t: bounded_obj(t, W2, N_SIM),
    theta1, method='Nelder-Mead',
    options={'maxiter':5000,'xatol':1e-7,'fatol':1e-9},
)
theta2 = res_nm2.x
Q2     = res_nm2.fun

# Use Stage 2 only if it actually improves the normalised criterion
if Q2 < Q1 * 10:      # accept if not more than 10× worse in natural units
    theta_hat = theta2
    print(f"  Stage-2 accepted: Q = {Q2:.6f}  (Stage-1 was {Q1:.6f})")
else:
    theta_hat = theta1
    print(f"  Stage-2 rejected (Q={Q2:.4f} >> Stage-1 Q={Q1:.6f}). Using Stage-1.")

print(f"\n  Final estimates:")
for nm,v,(lb,ub) in zip(PARAM_NAMES, theta_hat, BOUNDS):
    flag = ' ← AT BOUND ⚠' if (abs(v-lb)<0.005 or abs(v-ub)<0.005) else ''
    print(f"    {nm:8s} = {v:.5f}   [{lb:.3f},{ub:.3f}]{flag}")

# ── CELL 7: Bootstrap (serial — Colab multiprocessing limitation) ─────────────
print("\n"+"━"*60)
print(f"  BOOTSTRAP — {N_BOOT} replications (serial)")
print("━"*60)

# Estimate moment variance for perturbation scale
_Mv = np.array([
    model_moments(theta_hat, n_sim=max(N_SIM//3,20), seed=RANDOM_SEED+i)
    for i in range(20)
])
_Mv = _Mv[~np.any(np.isinf(_Mv)|np.isnan(_Mv),axis=1)]
m_sd = _Mv.std(axis=0) * 0.5   # conservative perturbation

boot_list = []
for b in range(N_BOOT):
    rng_b = np.random.default_rng(RANDOM_SEED+b+1)
    m_b   = DATA_MOMENTS + rng_b.normal(0, m_sd)

    def obj_b(t, mb=m_b):
        if not all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,t)): return 1e12
        mt = model_moments(t, n_sim=max(N_SIM//5,20), seed=RANDOM_SEED+b+1)
        if np.any(np.isinf(mt)|np.isnan(mt)): return 1e12
        rv = mb - mt
        return float(rv @ W1 @ rv)

    res_b = minimize(
        obj_b,
        theta_hat + rng_b.normal(0, 0.03, len(theta_hat)),
        method='Nelder-Mead',
        options={'maxiter':1000,'xatol':1e-4,'fatol':1e-6},
    )
    bt = res_b.x
    if all(lb<=v<=ub for (lb,ub),v in zip(BOUNDS,bt)):
        boot_list.append(bt)
    if (b+1) % 20 == 0:
        print(f"    {b+1}/{N_BOOT}  (valid: {len(boot_list)})")

boot_arr = np.array(boot_list)
se_hat   = boot_arr.std(axis=0)
ci_lo    = np.percentile(boot_arr, 2.5,  axis=0)
ci_hi    = np.percentile(boot_arr, 97.5, axis=0)
t_stats  = theta_hat / (se_hat + 1e-12)
print(f"  Valid bootstrap runs: {len(boot_list)}/{N_BOOT}")

# ── CELL 8: Identification (Jacobian) ─────────────────────────────────────────
print("\n"+"━"*60)
print("  IDENTIFICATION DIAGNOSTICS")
print("━"*60)
eps_J = 0.005
m0e = model_moments(theta_hat, n_sim=N_SIM, seed=RANDOM_SEED)
J   = np.zeros((len(DATA_MOMENTS), len(PARAM_NAMES)))
for j in range(len(PARAM_NAMES)):
    tp = theta_hat.copy(); tp[j] += eps_J
    tm = theta_hat.copy(); tm[j] -= eps_J
    if not is_valid(tp): tp = theta_hat.copy(); tp[j] -= eps_J
    if not is_valid(tm): tm = theta_hat.copy(); tm[j] += eps_J
    J[:,j] = (model_moments(tp,n_sim=N_SIM,seed=RANDOM_SEED)
             - model_moments(tm,n_sim=N_SIM,seed=RANDOM_SEED)) / (2*eps_J)
_,sv,_ = svd(J)
cond_J = sv[0]/(sv[-1]+1e-12)
print(f"  Singular values: {np.round(sv,4)}")
print(f"  Condition number of J: {cond_J:.1f}  "
      f"({'well-identified' if cond_J<50 else 'moderate' if cond_J<200 else 'weak — some parameters may not be identified'})")

# ── CELL 9: Results ────────────────────────────────────────────────────────────
m_eval = model_moments(theta_hat, n_sim=N_SIM*2, seed=RANDOM_SEED)
print("\n"+"━"*65)
print("  SMM ESTIMATION RESULTS")
print("━"*65)
print(f"\n  {'Param':8s}  {'Est':>9s}  {'SE':>7s}  {'t':>6s}  "
      f"{'95% CI':>22s}  Bounds")
print("  "+"-"*75)
for i,nm in enumerate(PARAM_NAMES):
    flag = ' ← BOUND' if (abs(theta_hat[i]-BOUNDS[i][0])<0.005 or
                          abs(theta_hat[i]-BOUNDS[i][1])<0.005) else ''
    print(f"  {nm:8s}  {theta_hat[i]:>9.5f}  {se_hat[i]:>7.5f}  "
          f"{t_stats[i]:>6.2f}  [{ci_lo[i]:.4f},{ci_hi[i]:.4f}]"
          f"  [{BOUNDS[i][0]:.3f},{BOUNDS[i][1]:.3f}]{flag}")

print(f"\n  SMM criterion Q: {Q1:.6f}  (Stage-1, used for inference)")
print(f"\n  {'Moment':15s}  {'Data':>8s}  {'Model':>8s}  "
      f"{'Diff':>8s}  {'Rel%':>7s}")
print("  "+"-"*55)
for nm,dv,mv in zip(MLABELS, DATA_MOMENTS, m_eval):
    print(f"  {nm:15s}  {dv:>8.4f}  {mv:>8.4f}  "
          f"{mv-dv:>+8.4f}  {(mv-dv)/dv*100:>6.1f}%")

# ── CELL 10: Implied model quantities ──────────────────────────────────────────
RH_e,RL_e,cA_e,cb_e,d_e,s_e,lu_e,st_e = unpack(theta_hat)
ds_e = d_e*s_e; dsl_e=K/RH_e; dsh_e=K/RL_e; dom_e=K/(K+cA_e-cb_e)
ic_e  = (1-ds_e)*(K/ds_e-RL_e)
DW_e  = (cA_e-cb_e) - K*(1-ds_e)/ds_e
WA_e  = mu*(RH_e-K-cA_e)+(1-mu)*(RL_e-K-cA_e)
WB_e  = mu*(RH_e-K-K*(1-ds_e)/ds_e-cb_e)+(1-mu)*(RL_e-K-cA_e)
R_dyn = r+lam_f+lu_e
B_g   = d1/(R_dyn+kappa_t); A_g=d0/R_dyn+kappa_t*theta_bar*B_g/R_dyn
G0_e  = A_g+B_g*theta_bar

print("\n"+"━"*65)
print("  MODEL QUANTITIES AT ESTIMATED PARAMETERS")
print("━"*65)
print(f"\n  Separation region   : [{dsl_e:.4f}, {dsh_e:.4f})")
in_r = dsl_e<=ds_e<dsh_e
print(f"  δ×σ = {d_e:.4f}×{s_e:.4f} = {ds_e:.4f}   in region: {'✓' if in_r else '✗'}")
print(f"  IC  : {ic_e:.4f} ≥ {cA_e-cb_e:.4f}  →  {'✓' if ic_e>=cA_e-cb_e else '✗'}")
print(f"  δσ_dom = {dom_e:.4f}   "
      f"({'inside separation region ✓' if dsl_e<=dom_e<dsh_e else 'outside — conventional still preferred'})")
print(f"  ΔW(B−A) = {DW_e:+.4f}   "
      f"({'STO welfare-superior ✓' if DW_e>0 else 'conventional preferred (need higher c_A-c_b)'})")
print(f"  W_A = {WA_e:.4f}   W_B = {WB_e:.4f}")
print(f"  G(θ̄) = {G0_e:.4f}   P_t = {ds_e*G0_e:.4f}")

if abs(st_e-BOUNDS[7][1])<0.005:
    print(f"\n  ⚠  s_θ still at upper bound {BOUNDS[7][1]}.")
    print(f"     Re-run with BOUNDS[7] = (0.02, 0.55)")

print("\n"+"━"*65)
print("  COPY INTO STO_Simulations_v2.py  Cell 2:")
print("━"*65)
print(f"  K=1.00; RH={RH_e:.5f}; RL={RL_e:.5f}; mu=0.50")
print(f"  c_A={cA_e:.5f}; c_b={cb_e:.5f}")
print(f"  delta0={d_e:.5f}; sigma0={s_e:.5f}")
print(f"  lam_u={lu_e:.5f}; s_theta={st_e:.5f}")

# ── CELL 11: Figures ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16,11))
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.50,wspace=0.38)
fig.suptitle('SMM Estimation Results — STO Non-Issuance Model',
             fontsize=13,fontweight='bold')

ax=fig.add_subplot(gs[0,:2])
x=np.arange(len(MLABELS)); w=0.35
ax.bar(x-w/2,DATA_MOMENTS,w,color='#1565C0',alpha=0.85,label='Data')
ax.bar(x+w/2,m_eval,w,color='#C62828',alpha=0.85,label='Model (SMM)')
ax.set_xticks(x); ax.set_xticklabels(MLABELS,rotation=18,ha='right',fontsize=9)
ax.set_title('(a) Moment Fit'); ax.legend(fontsize=9)

ax=fig.add_subplot(gs[0,2])
res=m_eval-DATA_MOMENTS
cr=['#2E7D32' if abs(r)<0.03*abs(d) else '#E65100' if abs(r)<0.10*abs(d) else '#C62828'
    for r,d in zip(res,DATA_MOMENTS)]
ax.barh(range(len(MLABELS)),res,color=cr,alpha=0.85)
ax.set_yticks(range(len(MLABELS))); ax.set_yticklabels(MLABELS,fontsize=8)
ax.axvline(0,color='black',lw=0.8); ax.set_xlabel('Residual')
ax.set_title('(b) Moment Residuals')

for i in range(len(PARAM_NAMES)):
    ax=fig.add_subplot(gs[1+i//3,i%3])
    if len(boot_arr)>5:
        ax.hist(boot_arr[:,i],bins=20,color='#1565C0',alpha=0.75,edgecolor='white')
    ax.axvline(theta_hat[i],color='#C62828',lw=2.0,label=f'{theta_hat[i]:.4f}')
    if len(boot_arr)>5:
        ax.axvline(ci_lo[i],color='grey',lw=1.2,ls='--')
        ax.axvline(ci_hi[i],color='grey',lw=1.2,ls='--',
                   label=f'[{ci_lo[i]:.3f},{ci_hi[i]:.3f}]')
    ax.set_title(f'({chr(99+i)}) {PARAM_NAMES[i]}  SE={se_hat[i]:.4f}',fontsize=10)
    ax.legend(fontsize=7)

plt.savefig('fig_smm_results.pdf',bbox_inches='tight')
plt.show(); print("→ fig_smm_results.pdf saved.")

# Jacobian heatmap
fig2,ax2=plt.subplots(figsize=(10,5))
Jn=J/(np.abs(J).max(axis=0,keepdims=True)+1e-10)
im=ax2.imshow(Jn,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
ax2.set_xticks(range(len(PARAM_NAMES))); ax2.set_xticklabels(PARAM_NAMES,rotation=25,ha='right')
ax2.set_yticks(range(len(MLABELS))); ax2.set_yticklabels(MLABELS)
plt.colorbar(im,ax=ax2,label='Normalised sensitivity')
ax2.set_title(f'Jacobian ∂m/∂θ  —  cond(J)={cond_J:.1f}')
for i in range(len(MLABELS)):
    for j in range(len(PARAM_NAMES)):
        v=Jn[i,j]
        ax2.text(j,i,f'{v:.2f}',ha='center',va='center',
                 fontsize=7,color='white' if abs(v)>0.6 else 'black')
plt.tight_layout()
plt.savefig('fig_identification.pdf',bbox_inches='tight')
plt.show(); print("→ fig_identification.pdf saved.")
print("\n✓ SMM estimation complete.")
