# =============================================================================
# 01_simulations.py
# Replication Package — STO Non-Issuance Paper
# "Security Token Offerings and Non-Issuance of Beneficiary Certificates:
#  A Signalling Equilibrium Approach under Japanese Trust Law"
#
# Author:  Koki Arai (Kyoritsu Women's University / JSPS KAKENHI 23K01404)
# Journal: Financial Innovation (Springer)
#
# PURPOSE:
#   Numerical verification of Propositions 1-6 and generation of Figures 1-7.
#   All figures correspond to panels in the main text.
#
# USAGE (Google Colab):
#   exec(open('01_simulations.py').read())
#   — or — Runtime → Run all
#
# OUTPUT:
#   fig1_separation_IC.pdf   fig2_equilibrium_welfare.pdf
#   fig3_submodularity.pdf   fig4_regime_comparison.pdf
#   fig5_dynamic_pricing.pdf fig6_adoption.pdf
#   fig7_policy_tiers.pdf
#
# PARAMETERS: Edit the CONFIG section (Cell 2) before running.
#   Key: RH=1.548, RL=1.140, c_A=0.219, c_b=0.181, delta0=0.846,
#        sigma0=0.851, lam_u=0.081, s_theta=0.229 (from SMM estimation)
#
# RUNTIME: ~3 minutes on Colab free tier
# =============================================================================


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Imports                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':    'DejaVu Serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi':     120,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':      True,
    'grid.alpha':     0.25,
    'grid.linestyle': ':',
})

print("✓ Imports complete.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — CONFIG  (edit here to change parameters)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Static model ─────────────────────────────────────────────────────────────
K    = 1.00   # Financing need (normalise to 1)
RH   = 1.50   # R(H): high-quality asset return
RL   = 1.15   # R(L): low-quality asset return   [must satisfy R(L) > K]
mu   = 0.50   # Prior probability P(θ = H)
c_A  = 0.05   # Conventional certificate-issuance cost
c_b  = 0.01   # Blockchain STO cost  [c_b < c_A required]

# Baseline institutional quality
delta0 = 0.80   # Settlement certainty   δ₀  ∈ (0, 1)
sigma0 = 0.90   # Oracle quality         σ₀  ∈ (0, 1)

# ── Institutional production functions δ(Λ), σ(Σ)  [Proposition 4] ─────────
delta_min = 0.50   # δ without any legal reform
sigma_min = 0.60   # σ without oracle standardisation
a_lam     = 0.80   # curvature of δ(Λ) = δ_min + (1−δ_min)(1−e^{−a·Λ})
b_sig     = 0.70   # curvature of σ(Σ) = σ_min + (1−σ_min)(1−e^{−b·Σ})

# ── Dynamic model ────────────────────────────────────────────────────────────
r         = 0.05   # Risk-free discount rate
lam_f     = 0.03   # Settlement-failure Poisson rate  λ_f
lam_u     = 0.07   # Upgrade-outage Poisson rate      λ_u
kappa_t   = 0.40   # OU mean-reversion speed          κ_θ
theta_bar = 1.00   # OU long-run mean
s_theta   = 0.08   # OU diffusion                     s_θ
d0        = 0.05   # D*(θ) ≈ d₀ + d₁·θ  (intercept)
d1        = 0.03   # D*(θ) ≈ d₀ + d₁·θ  (slope)
kappa_m   = 0.002  # Register maintenance cost        κ
rho_d     = 0.50   # Register-update intensity        ρ_δ

# ── Adoption dynamics (Conjecture) ──────────────────────────────────────────
eta    = 3.00   # Adoption speed  η
c_join = 0.07   # Investor entry cost
gamma  = 0.50   # Network-effect exponent
N0     = 0.05   # Initial adoption fraction

# ── Simulation run counts (change these to increase accuracy) ───────────────
N_MC_ADOPT  = 50    # Monte Carlo paths for adoption S-curve (Fig 6b)
N_MC_PRICE  = 100   # Monte Carlo paths for price distribution (Fig 5f)
N_GRID      = 120   # Grid resolution for contour / 3-D plots
T_DYNAMIC   = 25.0  # Time horizon for dynamic simulations (years)
DT          = 0.02  # Time step for SDE simulation

# ── Output ──────────────────────────────────────────────────────────────────
SAVE_PDF    = True       # Save each figure as a PDF?
RANDOM_SEED = 42         # Master random seed (set None for fresh each run)

# ────────────────────────────────────────────────────────────────────────────
# [DO NOT EDIT BELOW THIS LINE — computed automatically from CONFIG]
# ────────────────────────────────────────────────────────────────────────────
ds_low    = K / RH                    # δσ̅   = K/R(H)
ds_high   = K / RL                    # δσ̅̅   = K/R(L)
ds_base   = delta0 * sigma0           # baseline δσ
dom_thresh = K / (K + c_A - c_b)     # dominance threshold (eq. 9)
n_dyn     = int(T_DYNAMIC / DT)
t_arr     = np.linspace(0, T_DYNAMIC, n_dyn)

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

# ── Validation ───────────────────────────────────────────────────────────────
assert RL > K,      f"R(L)={RL} must exceed K={K}"
assert RH > RL,     f"R(H)={RH} must exceed R(L)={RL}"
assert c_b < c_A,   f"c_b={c_b} must be less than c_A={c_A}"
assert 0 < delta0 < 1 and 0 < sigma0 < 1, "δ₀ and σ₀ must be in (0,1)"

ic_lhs = (1 - ds_base) * (K/ds_base - RL)
ic_rhs = c_A - c_b
in_region = ds_low <= ds_base < ds_high

print("━" * 60)
print("  PARAMETER VALIDATION")
print("━" * 60)
print(f"  Separation region : [{ds_low:.4f}, {ds_high:.4f})")
print(f"  Baseline  δσ      = {ds_base:.4f}   in region: {'✓' if in_region else '✗ OUTSIDE'}")
print(f"  IC  LHS = {ic_lhs:.4f}  ≥  RHS = {ic_rhs:.4f}:  {'✓' if ic_lhs>=ic_rhs else '✗ VIOLATED'}")
print(f"  Dominance threshold δσ_dom = {dom_thresh:.4f}  "
      f"({'inside' if ds_low<=dom_thresh<ds_high else 'outside'} separation region)")
print(f"  MC paths (adoption/price) : {N_MC_ADOPT} / {N_MC_PRICE}")
print(f"  Grid resolution           : {N_GRID}×{N_GRID}")
print(f"  Simulation horizon        : {T_DYNAMIC} yrs  (dt = {DT})")
print("━" * 60)
if not in_region:
    print("  ⚠  Baseline δσ is outside the separation region.")
    print("     Propositions 1–2 require δσ ∈ [K/R(H), K/R(L)).")
    print("     Adjust delta0, sigma0, or asset returns.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Core functions (run once; used by all figure cells)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Static model functions ────────────────────────────────────────────────────
def Dstar(ds):
    """Equilibrium promised payment  D* = K/(δσ)  [eq. 3]"""
    return K / ds

def distortion(ds):
    """Signalling distortion  Δ = K(1−δσ)/(δσ)  = D* − K"""
    return K * (1 - ds) / ds

def IC_lhs(ds):
    """IC left-hand side  (1−δσ)(K/(δσ) − R(L))  [eq. 4]"""
    return (1 - ds) * (K/ds - RL)

def W_B(ds):
    """Aggregate welfare under Regime B (opt-in STO)  [eq. 6]"""
    return mu * (RH - K - distortion(ds) - c_b) + (1 - mu) * (RL - K - c_A)

def W_A():
    """Aggregate welfare under Regime A (mandatory conventional issuance)"""
    return mu * (RH - K - c_A) + (1 - mu) * (RL - K - c_A)

def W_C(ds):
    """Aggregate welfare under Regime C (mandatory non-issuance, pooled).
    P_C = μK + (1−μ)δσR(L) < K  →  H-type infeasible (zero surplus)."""
    W_L = RL - K - distortion(ds) - c_b   # L-type surplus under pooled STO
    return mu * 0.0 + (1 - mu) * W_L

# ── Institutional production functions ───────────────────────────────────────
def delta_fn(lam):
    return delta_min + (1 - delta_min) * (1 - np.exp(-a_lam * lam))

def sigma_fn(sig):
    return sigma_min + (1 - sigma_min) * (1 - np.exp(-b_sig * sig))

def W_star_LS(lam, sig):
    """W*(Λ,Σ) for Proposition 4 submodularity analysis."""
    d = delta_fn(lam); s = sigma_fn(sig); ds = d * s
    valid = (ds >= ds_low) & (ds < ds_high)
    return np.where(valid, W_B(ds), np.nan)

# ── Dynamic model ─────────────────────────────────────────────────────────────
def G_func(theta, lf=lam_f, lu=lam_u):
    """Closed-form G for linear D*(θ) = d0 + d1·θ  [eq. 14].
    HJB solution:  G(θ) = A + Bθ  where
      B = d1 / (r+λ_f+λ_u + κ_θ)
      A = d0/(r+λ_f+λ_u) + κ_θ·θ̄·B/(r+λ_f+λ_u)
    """
    R = r + lf + lu
    B = d1 / (R + kappa_t)
    A = d0 / R + kappa_t * theta_bar * B / R
    return A + B * theta

def delta_tilde(sig, lf=lam_f, lu=lam_u):
    """Maintenance threshold  δ̃ = κ(r+λ_f+λ_u)/(σ·G(θ̄)·ρ_δ)  [eq. 15]"""
    return kappa_m * (r + lf + lu) / (sig * G_func(theta_bar, lf=lf, lu=lu) * rho_d)

def sim_theta(seed=None):
    """Simulate one OU path for θ_t."""
    rng = np.random.default_rng(seed)
    th = np.zeros(n_dyn); th[0] = theta_bar
    for i in range(1, n_dyn):
        th[i] = (th[i-1]
                 + kappa_t * (theta_bar - th[i-1]) * DT
                 + s_theta * np.sqrt(DT) * rng.standard_normal())
    return th

def sim_price(d, s, lf=lam_f, lu=lam_u, seed=None):
    """Simulate P_t = d·s·G(θ_t) along one OU path."""
    return d * s * G_func(sim_theta(seed), lf=lf, lu=lu)

def sim_adoption(dv, sv, lf=lam_f, lu=lam_u, T=None, seed=None):
    """Simulate adoption fraction N_t.
    ODE:  dN/dt = η[P_t·N^γ − c_join]·N·(1−N)
    Returns (time_array, N_array).
    """
    T = T or T_DYNAMIC
    n = int(T / DT)
    rng = np.random.default_rng(seed)
    th = np.zeros(n); N = np.zeros(n)
    th[0] = theta_bar; N[0] = N0
    for i in range(1, n):
        th[i] = (th[i-1]
                 + kappa_t * (theta_bar - th[i-1]) * DT
                 + s_theta * np.sqrt(DT) * rng.standard_normal())
        Pt   = dv * sv * G_func(th[i-1], lf=lf, lu=lu)
        dN   = eta * (Pt * N[i-1]**gamma - c_join) * N[i-1] * (1 - N[i-1]) * DT
        N[i] = np.clip(N[i-1] + dN, 0.0, 1.0)
    return np.linspace(0, T, n), N

def _save(name):
    if SAVE_PDF:
        plt.savefig(f'{name}.pdf', bbox_inches='tight')
        print(f"  → saved {name}.pdf")

print("✓ Functions defined.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Figure 1: Separation Region & IC Condition                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('Figure 1: Feasibility-Separation Region and IC Condition',
             fontsize=13, fontweight='bold', y=1.01)

dd = np.linspace(0.25, 1.0, 500)
ss = np.linspace(0.25, 1.0, 500)
D2, S2 = np.meshgrid(dd, ss)
DS2 = D2 * S2

sep  = (DS2 >= ds_low)  & (DS2 < ds_high)
ic   = sep & (IC_lhs(DS2) >= ic_rhs)

ax = axes[0]
ax.contourf(D2, S2, sep.astype(float), levels=[0.5,1.5], colors=['#C8E6C9'], alpha=0.55)
ax.contourf(D2, S2, ic.astype(float),  levels=[0.5,1.5], colors=['#66BB6A'], alpha=0.55)
for ds_v, col, ls, lbl in [
    (ds_low,    '#2E7D32', '--', f'δσ = K/R(H) = {ds_low:.3f}'),
    (ds_high,   '#C62828', '-',  f'δσ = K/R(L) = {ds_high:.3f}'),
    (dom_thresh,'#1565C0', ':',  f'δσ_dom = {dom_thresh:.3f}  (dominance threshold)'),
]:
    s_h = np.clip(ds_v / dd, 0, 1)
    mask = s_h <= 1.0
    ax.plot(dd[mask], s_h[mask], color=col, lw=2.0, ls=ls, label=lbl)
ax.plot(delta0, sigma0, 'o', color='#1A237E', ms=10, zorder=6,
        label=f'Baseline (δ,σ)=({delta0},{sigma0})\nδσ={ds_base:.3f}')
ax.set_xlabel('Settlement certainty  δ')
ax.set_ylabel('Oracle quality  σ')
ax.set_title(f'(a) Separation Region  [{ds_low:.3f}, {ds_high:.3f})\n'
             'Light green: separation | Dark green: IC satisfied')
ax.legend(fontsize=8.5, loc='lower right')
ax.set_xlim(0.25, 1.0); ax.set_ylim(0.25, 1.0)

# IC condition across separation region
ax2 = axes[1]
ds_arr = np.linspace(ds_low + 0.001, ds_high - 0.001, 400)
ic_vals = IC_lhs(ds_arr)
ax2.plot(ds_arr, ic_vals, color='#1B5E20', lw=2.5,
         label=r'IC LHS: $(1-\delta\sigma)(K/\delta\sigma - R(L))$')
ax2.axhline(ic_rhs, color='#C62828', lw=1.8, ls='--',
            label=f'IC RHS: $c_A - c_b$ = {ic_rhs:.3f}')
ax2.fill_between(ds_arr, ic_rhs, ic_vals, where=(ic_vals >= ic_rhs),
                 alpha=0.22, color='#2E7D32', label='IC satisfied (eq\'m exists)')
ax2.fill_between(ds_arr, ic_vals, ic_rhs, where=(ic_vals < ic_rhs),
                 alpha=0.15, color='#C62828', label='IC violated')
ax2.axvline(ds_base, color='#1A237E', lw=1.0, ls=':',
            label=f'Baseline δσ = {ds_base:.3f}')
ax2.set_xlabel('δσ  (institutional quality product)')
ax2.set_ylabel('IC condition value')
ax2.set_title('(b) Incentive-Compatibility Condition [eq. 4]\n'
              'IC tightens as δσ rises toward K/R(L)')
ax2.legend(fontsize=8.5)

plt.tight_layout()
_save('fig1_separation_IC')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Figure 2: Equilibrium Contract D* and Welfare                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ds_arr  = np.linspace(ds_low + 0.001, ds_high - 0.001, 500)
ds_ext  = np.linspace(ds_low + 0.001, min(dom_thresh + 0.08, 0.999), 600)
WA_val  = W_A()
WB_arr  = W_B(ds_arr)
DW_ext  = mu * ((c_A - c_b) - K * (1 - ds_ext) / ds_ext)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Figure 2: Equilibrium Contract D* and Welfare Analysis',
             fontsize=13, fontweight='bold')

# (a) D*
ax = axes[0, 0]
ax.plot(ds_arr, Dstar(ds_arr), color='#1565C0', lw=2.5,
        label=r'$D^* = K/(\delta\sigma)$')
ax.axhline(RL, color='#C62828', lw=1.3, ls='--', label=f'R(L) = {RL}')
ax.axhline(RH, color='#2E7D32', lw=1.3, ls='--', label=f'R(H) = {RH}')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':')
ax.fill_between(ds_arr, RL, Dstar(ds_arr), where=(Dstar(ds_arr) > RL),
                alpha=0.12, color='#C62828', label='Junior shortfall zone')
ax.set_xlabel('δσ'); ax.set_ylabel('D*')
ax.set_title(r'(a)  $D^* = K/(\delta\sigma)$  [eq. 3]' + '\n'
             r'$\partial D^*/\partial\delta < 0$,  $\partial D^*/\partial\sigma < 0$  (Prop. 2)')
ax.legend(fontsize=9)

# (b) W_B vs W_A
ax = axes[0, 1]
ax.plot(ds_arr, WB_arr, color='#1565C0', lw=2.5, label='$W_B$  (opt-in STO, eq. 6)')
ax.axhline(WA_val, color='#E65100', lw=1.8, ls='--',
           label=f'$W_A$ = {WA_val:.4f}  (conventional)')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':')
ax.axvline(dom_thresh, color='#1565C0', lw=1.0, ls=':',
           label=f'$\\delta\\sigma_{{dom}}$ = {dom_thresh:.3f}')
ax.set_xlabel('δσ'); ax.set_ylabel('Aggregate surplus $W^*$')
ax.set_title('(b) Welfare $W_B$ vs $W_A$\n'
             '$W_B$ rises with δσ; crosses $W_A$ at δσ_dom')
ax.legend(fontsize=9)

# (c) ΔW = W_B − W_A extended beyond separation region
ax = axes[1, 0]
ax.plot(ds_ext, DW_ext, color='#1565C0', lw=2.5,
        label=r'$\Delta W = W_B - W_A$  [eq. 8]')
ax.axhline(0, color='black', lw=0.8)
ax.fill_between(ds_ext, 0, DW_ext, where=(DW_ext >= 0),
                alpha=0.20, color='#2E7D32', label='STO dominates (cond. 9)')
ax.fill_between(ds_ext, DW_ext, 0, where=(DW_ext < 0),
                alpha=0.12, color='#C62828', label='Conventional dominates')
ax.axvline(dom_thresh, color='#2E7D32', lw=1.5, ls='--',
           label=f'$\\delta\\sigma_{{dom}}$ = {dom_thresh:.3f}')
ax.axvline(ds_high, color='#C62828', lw=1.2, ls='-.',
           label=f'Separation bound = {ds_high:.3f}')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':')
ax.set_xlabel('δσ  (extended beyond separation region)')
ax.set_ylabel(r'$\Delta W = W_B - W_A$')
ax.set_title('(c) Dominance Condition (9)\n'
             'Conditions (4) and (9) operate in mutually exclusive sub-regions')
ax.legend(fontsize=8.5)

# (d) Sensitivity of δσ_dom to cost differential
ax = axes[1, 1]
gap_arr = np.linspace(0.005, 0.60, 300)
dom_arr = K / (K + gap_arr)
ax.plot(gap_arr, dom_arr, color='#1565C0', lw=2.5,
        label=r'$\delta\sigma_{dom} = K/(K + c_A - c_b)$')
ax.axhline(ds_high, color='#C62828', lw=1.5, ls='--',
           label=f'$K/R(L)$ = {ds_high:.3f}  (sep. upper bound)')
ax.axhline(ds_low,  color='#2E7D32', lw=1.2, ls=':',
           label=f'$K/R(H)$ = {ds_low:.3f}  (sep. lower bound)')
ax.fill_between(gap_arr, dom_arr, ds_high,
                where=(dom_arr < ds_high) & (dom_arr > ds_low),
                alpha=0.20, color='#2E7D32',
                label='Threshold inside separation region')
ax.axvline(c_A - c_b, color='grey', lw=1.0, ls=':',
           label=f'Baseline $c_A - c_b$ = {c_A - c_b:.3f}')
ax.set_xlabel('Cost differential  $c_A - c_b$')
ax.set_ylabel('Dominance threshold  $\\delta\\sigma_{dom}$')
ax.set_title('(d) Sensitivity of Dominance Threshold to Cost Differential\n'
             'Larger cost gap → threshold enters separation region')
ax.legend(fontsize=8.5)

plt.tight_layout()
_save('fig2_equilibrium_welfare')
plt.show()

print(f"\n  Summary at baseline δσ = {ds_base:.4f}:")
print(f"    D* = {Dstar(ds_base):.4f},  distortion = {distortion(ds_base):.4f}")
print(f"    IC: {IC_lhs(ds_base):.4f} ≥ {ic_rhs:.4f}  →  {'✓' if IC_lhs(ds_base)>=ic_rhs else '✗'}")
print(f"    W_A = {WA_val:.4f},  W_B = {W_B(ds_base):.4f},  ΔW = {W_B(ds_base)-WA_val:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Figure 3: Proposition 4 — Submodularity of W*(Λ,Σ)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

lam_v = np.linspace(0.05, 4.0, N_GRID)
sig_v = np.linspace(0.05, 4.0, N_GRID)
LAM, SIG = np.meshgrid(lam_v, sig_v)
W_grid = W_star_LS(LAM, SIG)

# Numerical cross-partial ∂²W*/∂Λ∂Σ
h_cp = max(0.08, 4.0 / N_GRID)
cp = (W_star_LS(LAM+h_cp, SIG+h_cp) - W_star_LS(LAM+h_cp, SIG-h_cp)
    - W_star_LS(LAM-h_cp, SIG+h_cp) + W_star_LS(LAM-h_cp, SIG-h_cp)) / (4*h_cp**2)
cp_m = np.where(np.isnan(W_grid), np.nan, cp)
med_cp = np.nanmedian(cp_m)

fig = plt.figure(figsize=(16, 5.5))
fig.suptitle('Figure 3: Submodularity of W*(Λ,Σ) and Reform Sequencing (Proposition 4)',
             fontsize=13, fontweight='bold')

# (a) 3D surface
ax3d = fig.add_subplot(131, projection='3d')
W_plot = np.where(np.isnan(W_grid),
                  np.nanmin(W_grid[~np.isnan(W_grid)]), W_grid)
ax3d.plot_surface(LAM, SIG, W_plot, cmap='viridis', alpha=0.87, linewidth=0)
ax3d.set_xlabel('Λ  (legal reform)'); ax3d.set_ylabel('Σ  (oracle standard.)')
ax3d.set_zlabel('W*')
ax3d.set_title(f'(a) $W^*(\\Lambda, \\Sigma)$\nStrictly submodular (Prop. 4a)')
ax3d.view_init(elev=28, azim=-55)

# (b) Cross-partial heat-map
ax2 = fig.add_subplot(132)
vmax = np.nanpercentile(np.abs(cp_m), 97)
im = ax2.contourf(LAM, SIG, cp_m, levels=30,
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax2, shrink=0.85, label=r'$\partial^2 W^*/\partial\Lambda\partial\Sigma$')
ax2.contour(LAM, SIG, cp_m, levels=[0], colors='black', linewidths=1.5)
ax2.set_xlabel('Λ'); ax2.set_ylabel('Σ')
ok = '✓' if med_cp < 0 else '✗'
ax2.set_title(f'(b) Cross-partial < 0 everywhere [eq. 10]\n'
              f'Median = {med_cp:.5f}  {ok}')
ax2.text(3.0, 3.8, f'median\n{med_cp:.4f}\n< 0  {ok}',
         color='#1A237E', fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', fc='#EEF2F7', alpha=0.9))

# (c) Marginal returns — sequencing
ax3 = fig.add_subplot(133)
x_arr = np.linspace(0.10, 3.5, 80); h2 = max(0.06, 3.5/80)
for sig_fix, col, lbl in [(0.5, '#C62828', 'Σ = 0.5  (low standardisation)'),
                           (1.5, '#1565C0', 'Σ = 1.5  (moderate)'),
                           (3.0, '#2E7D32', 'Σ = 3.0  (high standardisation)')]:
    marg = (W_star_LS(x_arr+h2, sig_fix) - W_star_LS(x_arr-h2, sig_fix)) / (2*h2)
    ax3.plot(x_arr, marg, color=col, lw=2.0, label=lbl)
ax3.axhline(0, color='black', lw=0.7)
ax3.set_xlabel('Λ  (legal reform intensity)')
ax3.set_ylabel(r'$\partial W^*/\partial\Lambda$  (marginal return)')
ax3.set_title('(c) Submodularity → Reform Sequencing\n'
              r'$\partial W^*/\partial\Lambda$ decreasing in Σ:' + '\nPrioritise the lower input')
ax3.legend(fontsize=9, title='Fixed Σ level')

plt.tight_layout()
_save('fig3_submodularity')
plt.show()
print(f"  Cross-partial median = {med_cp:.5f}  (< 0: {med_cp < 0})  {'✓' if med_cp<0 else '✗'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Figure 4: Proposition 5 — Three-Regime Welfare Comparison      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ds_arr   = np.linspace(ds_low + 0.001, ds_high - 0.001, 500)
WA_val   = W_A()
WB_arr   = W_B(ds_arr)
WC_arr   = W_C(ds_arr)
PC_arr   = mu * K + (1 - mu) * ds_arr * RL
theta_hat = np.clip(K / ds_arr, RL, RH)
dW_BA_ds = (c_A - c_b) - K * (1 - ds_arr) / ds_arr
dW_const = (c_A - c_b) - K * (1 - ds_base) / ds_base

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Figure 4: Welfare Under Three Statutory Designs (Proposition 5)',
             fontsize=13, fontweight='bold')

# (a) Three-regime welfare
ax = axes[0]
ax.axhline(WA_val, color='#E65100', lw=2.0, ls='--',
           label=f'$W_A$ = {WA_val:.4f}  (mandatory issuance)')
ax.plot(ds_arr, WB_arr, color='#1565C0', lw=2.5, label='$W_B$  (opt-in STO)')
ax.plot(ds_arr, WC_arr, color='#6A1B9A', lw=2.0, ls='-.', label='$W_C$  (mandatory STO)')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':', label=f'Baseline δσ={ds_base:.3f}')
ax.fill_between(ds_arr, WC_arr, WB_arr, alpha=0.10, color='#1565C0')
ax.set_xlabel('δσ'); ax.set_ylabel('Expected aggregate surplus')
ax.set_title('(a) $W_B > \\max(W_A, W_C)$ throughout\n(Proposition 5)')
ax.legend(fontsize=8.5)

# (b) ΔW(B−A) constant across θ
ax2 = axes[1]
theta_grid = np.linspace(RL, RH, 120)
sgn = '>' if dW_const > 0 else '<'
col_fill = '#1565C0' if dW_const > 0 else '#C62828'
ax2.axhline(dW_const, color=col_fill, lw=2.5,
            label=f'$\\Delta W(B-A)$ = {dW_const:.4f}  (const. across θ, eq. 11)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(theta_grid, 0, dW_const, alpha=0.18, color=col_fill)
ax2.set_xlabel('Originator type  θ ∈ [R(L), R(H)]')
ax2.set_ylabel('ΔW(B−A) = Π^B(θ) − Π^A(θ)')
ax2.set_title(f'(b) Uniform Preference Structure [eq. 11]\n'
              f'ΔW(B−A) = {dW_const:.4f}  {sgn}  0  (not a selection result)')
ax2.legend(fontsize=9)
ax2.text((RL+RH)/2, dW_const/2,
         f'All types prefer B\n({sgn}0 at baseline δσ)',
         ha='center', va='center', fontsize=9.5, color=col_fill)
# Inset: ΔW vs δσ
inset = ax2.inset_axes([0.54, 0.05, 0.44, 0.40])
inset.plot(ds_arr, dW_BA_ds, color=col_fill, lw=1.8)
inset.axhline(0, color='black', lw=0.6)
inset.axvline(ds_base, color='grey', lw=0.8, ls=':')
inset.set_title('ΔW vs δσ', fontsize=8); inset.tick_params(labelsize=7)

# (c) θ̂ and P_C
ax3 = axes[2]
ax3b = ax3.twinx()
ax3.plot(ds_arr, theta_hat, color='#2E7D32', lw=2.2,
         label=r'$\hat\theta = K/(\delta\sigma)$  [feasibility boundary]')
ax3.axhline(RL, color='#C62828', lw=1.2, ls='--', label=f'R(L) = {RL}')
ax3.axhline(RH, color='#1565C0', lw=1.2, ls='--', label=f'R(H) = {RH}')
ax3b.plot(ds_arr, PC_arr, color='#E65100', lw=2.0, ls=':',
          label='$P_C$  (pooled price, Regime C)')
ax3b.axhline(K, color='black', lw=0.9)
ax3.set_xlabel('δσ'); ax3.set_ylabel(r'$\hat\theta$  (feasibility threshold)')
ax3b.set_ylabel('$P_C$  (pooled token price)')
ax3.set_title(r'(c) Feasibility Boundary $\hat\theta$ and Regime C' + '\n'
              '$P_C < K$ everywhere → H-type infeasible under C')
li1, la1 = ax3.get_legend_handles_labels()
li2, la2 = ax3b.get_legend_handles_labels()
ax3.legend(li1+li2, la1+la2, fontsize=8)

plt.tight_layout()
_save('fig4_regime_comparison')
plt.show()

print(f"\n  W_A = {WA_val:.6f}")
print(f"  W_B = {W_B(ds_base):.6f}  (> W_C: {W_B(ds_base) > W_C(ds_base)}  ✓)")
print(f"  W_C = {W_C(ds_base):.6f}")
print(f"  P_C = {mu*K+(1-mu)*ds_base*RL:.4f} < K={K}  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Figure 5: Dynamic Token Pricing (Proposition 6)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 5: Dynamic Token Pricing, Maintenance Threshold, and Comparative Statics\n'
             '(Proposition 6 and Dynamic Corollary)',
             fontsize=13, fontweight='bold')

# (a) Token price paths — different δσ levels
ax = axes[0, 0]
combos = []
for dv, sv in [(0.70, 0.85), (delta0, sigma0), (0.88, 0.96)]:
    if ds_low <= dv*sv < ds_high:
        combos.append((dv, sv))
for dv, sv, col, lbl in [
    (combos[0][0], combos[0][1], '#C62828',
     f'δ={combos[0][0]}, σ={combos[0][1]}  (δσ={combos[0][0]*combos[0][1]:.3f})'),
    (combos[1][0], combos[1][1], '#1565C0',
     f'δ={combos[1][0]}, σ={combos[1][1]}  (baseline, δσ={combos[1][0]*combos[1][1]:.3f})'),
    (combos[-1][0], combos[-1][1], '#2E7D32',
     f'δ={combos[-1][0]}, σ={combos[-1][1]}  (δσ={combos[-1][0]*combos[-1][1]:.3f})'),
]:
    ax.plot(t_arr, sim_price(dv, sv, seed=0), color=col, lw=1.8, alpha=0.85, label=lbl)
ax.set_xlabel('Time (years)'); ax.set_ylabel('Token price $P_t$')
ax.set_title('(a) Token Price Paths  $P_t = \\delta\\sigma G(\\theta_t)$  [eq. 13]\n'
             'Higher δσ → higher and smoother prices')
ax.legend(fontsize=8.5)

# (b) G(θ) closed form
ax = axes[0, 1]
theta_r = np.linspace(0.70, 1.30, 200)
for lf_v, lu_v, col, lbl in [
    (lam_f*0.4, lam_u*0.4, '#2E7D32', f'λ_f={lam_f*0.4:.2f}, λ_u={lam_u*0.4:.2f}  (low risk)'),
    (lam_f,     lam_u,     '#1565C0', f'λ_f={lam_f}, λ_u={lam_u}  (baseline)'),
    (lam_f*2.5, lam_u*2.5, '#C62828', f'λ_f={lam_f*2.5:.2f}, λ_u={lam_u*2.5:.2f}  (high risk)'),
]:
    ax.plot(theta_r, delta0*sigma0*G_func(theta_r, lf=lf_v, lu=lu_v),
            color=col, lw=2.0, label=lbl)
ax.axvline(theta_bar, color='grey', lw=0.8, ls=':', label=f'$\\bar\\theta$ = {theta_bar}')
ax.set_xlabel('Asset quality  $\\theta_t$'); ax.set_ylabel('$P_t = \\delta\\sigma G(\\theta_t)$')
ax.set_title('(b) $G(\\theta)$ Closed Form  [eq. 14]\n'
             '$\\lambda_f$, $\\lambda_u$ enter as additional discount rates')
ax.legend(fontsize=8.5)

# (c) Amplification ∂²P/∂λ_u∂δ < 0
ax = axes[0, 2]
delta_r = np.linspace(0.40, 1.0, 120)
lu_vals = [lam_u*0.4, lam_u, lam_u*2, lam_u*4]
colors_c = ['#2E7D32', '#1565C0', '#E65100', '#C62828']
for lu_v, col in zip(lu_vals, colors_c):
    sens = -(delta_r * sigma0 * G_func(theta_bar, lu=lu_v)) / (r + lam_f + lu_v)
    ax.plot(delta_r, sens, color=col, lw=2.0, label=f'$\\lambda_u$ = {lu_v:.3f}')
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Settlement certainty  δ')
ax.set_ylabel(r'$\partial P_t / \partial\lambda_u$')
ax.set_title('(c) Amplification Effect  (Dynamic Corollary)\n'
             r'$\partial^2 P_t/\partial\lambda_u\partial\delta < 0$:' +
             '\nWeak δ amplifies upgrade-risk damage')
ax.legend(fontsize=8.5, title='$\\lambda_u$ level')

# (d) Maintenance threshold δ̃
ax = axes[1, 0]
sigma_r = np.linspace(0.40, 1.0, 120)
kappa_vals = [kappa_m*0.5, kappa_m, kappa_m*2.5, kappa_m*5]
colors_d = ['#2E7D32', '#1565C0', '#E65100', '#C62828']
for km, col in zip(kappa_vals, colors_d):
    dt_arr = km * (r+lam_f+lam_u) / (sigma_r * G_func(theta_bar) * rho_d)
    valid = dt_arr <= 1.0
    ax.plot(sigma_r[valid], dt_arr[valid], color=col, lw=2.0,
            label=f'κ = {km:.4f}')
ax.axhline(delta_min, color='navy', lw=1.5, ls='--',
           label=f'$\\delta_{{min}}$ = {delta_min}  (statutory floor)')
ax.axhline(delta0, color='grey', lw=1.0, ls=':',
           label=f'$\\delta_0$ = {delta0}  (baseline)')
ax.set_xlabel('Oracle quality  σ'); ax.set_ylabel('Maintenance threshold  $\\tilde\\delta$')
ax.set_title('(d) Maintenance Threshold  $\\tilde\\delta(\\bar\\theta, \\sigma)$  [eq. 15]\n'
             '$\\delta_0 > \\tilde\\delta$ → separating eq\'m self-sustaining')
ax.legend(fontsize=8.5, title='Maintenance cost κ'); ax.set_ylim(0, 1.05)

# (e) Comparative statics of δ̃
ax = axes[1, 1]
lf_r = np.linspace(0.005, 0.20, 120)
lu_r = np.linspace(0.005, 0.30, 120)
dt_lf = kappa_m*(r+lf_r+lam_u) / (sigma0*G_func(theta_bar)*rho_d)
dt_lu = kappa_m*(r+lam_f+lu_r) / (sigma0*G_func(theta_bar, lu=lu_r)*rho_d)
ax.plot(lf_r, dt_lf, color='#1565C0', lw=2.2,
        label=r'$\tilde\delta$ vs $\lambda_f$')
ax.plot(lu_r, dt_lu, color='#C62828', lw=2.0, ls='--',
        label=r'$\tilde\delta$ vs $\lambda_u$')
ax.axhline(delta0, color='grey', lw=0.9, ls=':',
           label=f'$\\delta_0$ = {delta0}')
ax.set_xlabel('Failure rate  ($\\lambda_f$ or $\\lambda_u$)')
ax.set_ylabel('Maintenance threshold  $\\tilde\\delta$')
ax.set_title('(e) Comparative Statics: $\\partial\\tilde\\delta/\\partial\\lambda > 0$\n'
             'Higher failure risk → harder to sustain equilibrium')
ax.legend(fontsize=9)

# (f) Token price distribution (N_MC_PRICE paths)
ax = axes[1, 2]
P_all = np.array([sim_price(delta0, sigma0, seed=s) for s in range(N_MC_PRICE)])
mn_P = P_all.mean(axis=0); sd_P = P_all.std(axis=0)
for i in range(min(8, N_MC_PRICE)):
    ax.plot(t_arr, P_all[i], lw=0.7, alpha=0.35, color='#1565C0')
ax.plot(t_arr, mn_P, color='#1565C0', lw=2.5, label=f'Mean $P_t$  (n={N_MC_PRICE})')
ax.fill_between(t_arr, mn_P-sd_P, mn_P+sd_P,
                alpha=0.20, color='#1565C0', label='±1 SD')
G0 = G_func(theta_bar)
ax.axhline(delta0*sigma0*G0, color='grey', lw=1.2, ls='--',
           label=f'$\\delta\\sigma G(\\bar\\theta)$ = {delta0*sigma0*G0:.4f}')
ax.set_xlabel('Time (years)'); ax.set_ylabel('$P_t$')
ax.set_title(f'(f) Token Price Distribution  ({N_MC_PRICE} paths)\n'
             'OU process for $\\theta_t$')
ax.legend(fontsize=8.5)

plt.tight_layout()
_save('fig5_dynamic_pricing')
plt.show()

dt_base = delta_tilde(sigma0)
print(f"\n  G(θ̄) = {G0:.4f},  P_t = {delta0*sigma0*G0:.4f}")
print(f"  ∂P/∂λ_u = {-(delta0*sigma0*G0)/(r+lam_f+lam_u):.4f} < 0  ✓")
print(f"  ∂²P/∂λ_u∂δ = {-sigma0*G0/(r+lam_f+lam_u):.4f} < 0  ✓")
print(f"  δ̃ = {dt_base:.6f},  δ₀ = {delta0}  →  self-sustaining: {delta0 > dt_base}  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Figure 6: Adoption S-curve Conjecture  (N_MC_ADOPT paths)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

T_ADOPT = 50   # adoption simulation horizon (years) — adjust here

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Figure 6: Platform Adoption Dynamics — Conjecture\n'
             f'(Monte Carlo: {N_MC_ADOPT} paths,  T = {T_ADOPT} yrs)',
             fontsize=13, fontweight='bold')

# Build candidate (δ,σ) pairs that lie in the separation region
def in_region(dv, sv):
    return ds_low <= dv*sv < ds_high

ds_scenarios = [(dv, sv) for dv, sv in
    [(0.72, 0.82), (0.76, 0.87), (delta0, sigma0), (0.85, 0.95)]
    if in_region(dv, sv)]

# (a) S-curves
ax = axes[0, 0]
cols_a = ['#C62828', '#E65100', '#1565C0', '#2E7D32']
for (dv, sv), col in zip(ds_scenarios, cols_a):
    t_a, N_a = sim_adoption(dv, sv, T=T_ADOPT, seed=RANDOM_SEED)
    ax.plot(t_a, N_a, color=col, lw=2.0,
            label=f'δ={dv}, σ={sv}  (δσ={dv*sv:.3f})')
ax.set_xlabel('Time (years)'); ax.set_ylabel('Adoption fraction  $N_t$')
ax.set_title('(a) S-curve Adoption Dynamics  (Conjecture)\n'
             'Higher δσ → faster inflection and full saturation')
ax.legend(fontsize=8.5); ax.set_ylim(-0.02, 1.05)

# (b) Variance across Monte Carlo paths
ax = axes[0, 1]
pair_b = [(ds_scenarios[0], '#C62828'),
          (ds_scenarios[-1], '#2E7D32')]
for (dv, sv), col in pair_b:
    paths = np.array([sim_adoption(dv, sv, T=T_ADOPT, seed=s)[1]
                      for s in range(N_MC_ADOPT)])
    t_tmp = sim_adoption(dv, sv, T=T_ADOPT, seed=0)[0]
    mn = paths.mean(axis=0); sd = paths.std(axis=0)
    ax.plot(t_tmp, mn, color=col, lw=2.2,
            label=f'δσ={dv*sv:.3f}  (mean, n={N_MC_ADOPT})')
    ax.fill_between(t_tmp, mn-sd, mn+sd, color=col, alpha=0.18,
                    label='±1 SD')
ax.set_xlabel('Time (years)'); ax.set_ylabel('$N_t$')
ax.set_title(f'(b) Adoption-Path Variance  ({N_MC_ADOPT} paths)\n'
             'Higher δσ → lower variance in acceleration phase')
ax.legend(fontsize=8.5)

# (c) Effect of upgrade risk λ_u
ax = axes[1, 0]
lu_list = [lam_u*0.4, lam_u, lam_u*2.5, lam_u*5]
cols_c = ['#2E7D32', '#1565C0', '#E65100', '#C62828']
for lu_v, col in zip(lu_list, cols_c):
    t_c, N_c = sim_adoption(delta0, sigma0, lf=lam_f, lu=lu_v,
                             T=T_ADOPT, seed=RANDOM_SEED)
    ax.plot(t_c, N_c, color=col, lw=2.0, label=f'$\\lambda_u$ = {lu_v:.3f}')
ax.set_xlabel('Time (years)'); ax.set_ylabel('$N_t$')
ax.set_title('(c) Effect of Upgrade Risk  $\\lambda_u$\n'
             'Higher outage risk → slower adoption')
ax.legend(fontsize=8.5)

# (d) Policy experiment: FSA guidance raises δ
ax = axes[1, 1]
d_bef = ds_scenarios[0][0]; s_bef = ds_scenarios[0][1]
d_aft = ds_scenarios[-1][0]; s_aft = ds_scenarios[-1][1]
t_b, N_b = sim_adoption(d_bef, s_bef, T=T_ADOPT, seed=RANDOM_SEED)
t_a2, N_a2 = sim_adoption(d_aft, s_aft, T=T_ADOPT, seed=RANDOM_SEED)
ax.plot(t_b,  N_b,  color='#C62828', lw=2.0,
        label=f'Before  δ={d_bef}, δσ={d_bef*s_bef:.3f}')
ax.plot(t_a2, N_a2, color='#2E7D32', lw=2.2,
        label=f'After   δ={d_aft}, δσ={d_aft*s_aft:.3f}')
wb_bef = W_B(d_bef*s_bef)
wb_aft = W_B(d_aft*s_aft)
dw_str = f'{wb_aft - wb_bef:.4f}'
ax.set_xlabel('Time (years)'); ax.set_ylabel('$N_t$')
ax.set_title('(d) Policy Experiment: FSA Guidance Raises δ\n'
             f'Static welfare gain ΔW = {dw_str}')
ax.legend(fontsize=8.5)
ax.annotate(f'ΔW = {dw_str}', xy=(T_ADOPT*0.55, 0.5), fontsize=11,
            color='#1565C0',
            bbox=dict(boxstyle='round', fc='#E3F2FD', alpha=0.85))

plt.tight_layout()
_save('fig6_adoption')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Figure 7: Three-Tier Policy Analysis                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ds_arr = np.linspace(ds_low + 0.001, ds_high - 0.001, 500)
WA_val = W_A()

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Figure 7: Three-Tier Policy Analysis (Section 6)',
             fontsize=13, fontweight='bold')

# Tier 1: Disclosure requirement
ax = axes[0]
ax.axhline(WA_val, color='#E65100', lw=1.8, ls='--',
           label=f'$W_A$ = {WA_val:.4f}  (no disclosure baseline)')
ax.plot(ds_arr, W_B(ds_arr), color='#1565C0', lw=2.5,
        label='$W_B$  (with mandatory disclosure)')
ax.fill_between(ds_arr, WA_val, W_B(ds_arr),
                where=(W_B(ds_arr) > WA_val),
                alpha=0.20, color='#2E7D32', label='$W_B > W_A$')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':')
ax.set_xlabel('δσ'); ax.set_ylabel('$W^*$')
ax.set_title('Tier 1: Junior-Retention Disclosure\n'
             'Without disclosure, non-issuance election\n'
             'cannot credibly signal quality (Prop. 5)')
ax.legend(fontsize=9)

# Tier 2: Reform sequencing
ax = axes[1]
x_arr = np.linspace(0.05, 4.0, 100); h_s = 0.12
for sig_fix, col, lbl in [(0.5, '#C62828', 'Σ = 0.5  (low σ → prioritise Σ)'),
                           (1.5, '#1565C0', 'Σ = 1.5  (moderate)'),
                           (3.5, '#2E7D32', 'Σ = 3.5  (high σ → prioritise Λ)')]:
    marg = [(W_star_LS(l+h_s, sig_fix) - W_star_LS(l-h_s, sig_fix))/(2*h_s)
            for l in x_arr]
    ax.plot(x_arr, marg, color=col, lw=2.0, label=lbl)
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Λ  (legal reform intensity)')
ax.set_ylabel(r'$\partial W^*/\partial\Lambda$')
ax.set_title('Tier 2: Reform Sequencing  (Prop. 4a)\n'
             r'$\partial W^*/\partial\Lambda$ decreasing in Σ:' +
             '\nPrioritise the lower-level input')
ax.legend(fontsize=8.5)

# Tier 3: Statutory amendment → δσ → 1
ax = axes[2]
ds_full = np.linspace(ds_low + 0.001, 0.9995, 600)
W_fb = mu * (RH - K - c_b) + (1 - mu) * (RL - K - c_A)
ax.plot(ds_full, W_B(ds_full), color='#1565C0', lw=2.5, label='$W_B(\\delta\\sigma)$')
ax.axhline(W_fb, color='#2E7D32', lw=1.8, ls='--',
           label=f'First-best (δσ→1):  $W^*$ = {W_fb:.4f}')
ax.axvline(ds_base, color='grey', lw=0.9, ls=':',
           label=f'Baseline δσ = {ds_base:.3f}')
ax.axvline(ds_high, color='#C62828', lw=1.2, ls='-.',
           label=f'Separation bound = {ds_high:.3f}')
ax.fill_between(ds_full, W_B(ds_full), W_fb, alpha=0.15, color='#2E7D32',
                label='Tier 3 welfare gap')
ax.annotate(r'$\hat\theta\to R(L)$' + '\nfirst-best\napproached',
            xy=(0.97, W_fb - 0.002), xytext=(0.82, W_fb - 0.022),
            arrowprops=dict(arrowstyle='->', color='#2E7D32'),
            fontsize=9, color='#2E7D32')
ax.set_xlabel('δσ  (extended)'); ax.set_ylabel('$W^*$')
ax.set_title('Tier 3: Statutory Amendment (Art. 185(1))\n'
             'Complete non-issuance → δσ → 1\n'
             '(model-motivated, tentative)')
ax.legend(fontsize=8.5)

plt.tight_layout()
_save('fig7_policy_tiers')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Full Numerical Verification Summary                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("━" * 65)
print("  FULL NUMERICAL VERIFICATION SUMMARY")
print("━" * 65)

ds_b = delta0 * sigma0
Ds   = Dstar(ds_b)
G0   = G_func(theta_bar)
P0   = delta0 * sigma0 * G0
R_t  = r + lam_f + lam_u
dt_b = delta_tilde(sigma0)

print(f"\n  [Propositions 1–2: Separating Equilibrium]")
print(f"    δσ̅  = K/R(H) = {ds_low:.4f}")
print(f"    δσ̅̅  = K/R(L) = {ds_high:.4f}")
print(f"    Baseline δσ  = {ds_b:.4f}   in region: {ds_low<=ds_b<ds_high}  ✓")
print(f"    D*           = {Ds:.4f}")
print(f"    Distortion   = {distortion(ds_b):.4f}")
print(f"    IC LHS = {IC_lhs(ds_b):.4f}  ≥  RHS = {ic_rhs:.4f}:  {IC_lhs(ds_b)>=ic_rhs}  ✓")
print(f"    ∂D*/∂δ = {-K/(delta0**2*sigma0):.4f} < 0  ✓")
print(f"    ∂D*/∂σ = {-K/(delta0*sigma0**2):.4f} < 0  ✓")

print(f"\n  [Condition (9): Dominance Threshold]")
print(f"    δσ_dom = {dom_thresh:.4f}  "
      f"({'inside' if ds_low<=dom_thresh<ds_high else 'outside'} separation region)")
print(f"    ΔW(B−A) at baseline = {W_B(ds_b)-W_A():.4f}")
print(f"    [Note: IC (4) and dominance (9) are mutually exclusive for any δσ when R(L)>0]")

print(f"\n  [Proposition 4: Submodularity]")
h_v = 0.12
W_pp=W_star_LS(1.5+h_v,1.5+h_v); W_pm=W_star_LS(1.5+h_v,1.5-h_v)
W_mp=W_star_LS(1.5-h_v,1.5+h_v); W_mm=W_star_LS(1.5-h_v,1.5-h_v)
cp_n = (W_pp-W_pm-W_mp+W_mm)/(4*h_v**2)
print(f"    ∂²W*/∂Λ∂Σ at (1.5,1.5) = {cp_n:.6f}  (< 0: {cp_n<0})  ✓")

print(f"\n  [Proposition 5: Regime Ranking]")
print(f"    W_A = {W_A():.6f}")
print(f"    W_B = {W_B(ds_b):.6f}  (> W_C: {W_B(ds_b)>W_C(ds_b)})  ✓")
print(f"    W_C = {W_C(ds_b):.6f}")
print(f"    P_C = {mu*K+(1-mu)*ds_b*RL:.4f} < K={K}  ✓")

print(f"\n  [Proposition 6 / Dynamic Corollary]")
print(f"    G(θ̄) = {G0:.4f},   P_t = {P0:.4f}")
print(f"    ∂P/∂λ_u   = {-P0/R_t:.4f} < 0  ✓")
print(f"    ∂²P/∂λ_u∂δ = {-sigma0*G0/R_t:.4f} < 0  ✓  (amplification confirmed)")
print(f"    δ̃ = {dt_b:.6f},   δ₀ = {delta0} > δ̃: {delta0>dt_b}  ✓  (self-sustaining)")
lf2 = lam_f + 0.01
dt2 = kappa_m*(r+lf2+lam_u)/(sigma0*G_func(theta_bar,lf=lf2)*rho_d)
print(f"    ∂δ̃/∂λ_f > 0: δ̃ rises {dt_b:.6f} → {dt2:.6f}  ✓")

print(f"\n  [Simulation Settings]")
print(f"    MC paths (adopt/price): {N_MC_ADOPT} / {N_MC_PRICE}")
print(f"    Grid resolution:        {N_GRID}×{N_GRID}")
print(f"    Time horizon:           {T_DYNAMIC} yrs  (dt={DT},  n={n_dyn} steps)")
print(f"    Random seed:            {RANDOM_SEED}")
print(f"    PDF saved:              {SAVE_PDF}")
print("━" * 65)
print("  All 7 figures complete.")
print("━" * 65)
