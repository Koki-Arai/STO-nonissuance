# =============================================================================
# 05_sensitivity_analysis.py
# Replication Package — STO Non-Issuance Paper
# "Security Token Offerings and Non-Issuance of Beneficiary Certificates:
#  A Signalling Equilibrium Approach under Japanese Trust Law"
#
# Author:  Koki Arai (Kyoritsu Women's University / JSPS KAKENHI 23K01404)
# Journal: Financial Innovation (Springer)
#
# PURPOSE:
#   Online Appendix OA.3: Welfare sensitivity to perturbation of delta*sigma.
#   Tables OA.4 (delta*sigma grid), OA.5 (cost structure), OA.6 (asset returns).
#
# KEY RESULTS (Table OA.4):
#   delta*sigma    W_A     W_B      ΔW(B-A)   Ranking
#   0.720 (base)   0.125  -0.051   -0.176    A > B > C
#   0.770          0.125  -0.005   -0.130    A > B > C
#   0.820          0.125  +0.034   -0.091    A > B > C  (+0.10 perturbation)
#
#   Full reversal W_B > W_A requires delta*sigma >= delta*sigma_dom = 0.963.
#   W_B > W_C throughout the separation region (Proposition 5, part ii).
#
# USAGE (Google Colab):
#   exec(open('05_sensitivity_analysis.py').read())
#
# RUNTIME: ~3 minutes on Colab free tier
# =============================================================================

# ── CELL 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.titlesize': 12, 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
})
print("✓ Imports OK.")

# ── CELL 2: Baseline parameters (SMM estimates, Appendix B) ───────────────────
K=1.00; mu=0.50
RH=1.548; RL=1.140
c_A=0.219; c_b=0.181
delta0=0.846; sigma0=0.851
ds_base = delta0 * sigma0     # 0.7204
RANDOM_SEED = 42

print(f"  SMM baseline: δσ = {ds_base:.4f}")
print(f"  Sep. region: [{K/RH:.4f}, {K/RL:.4f})")
print(f"  δσ_dom = {K/(K+c_A-c_b):.4f}")

# ── CELL 3: Welfare functions ─────────────────────────────────────────────────
def W_A(RH=RH, RL=RL, cA=c_A, cb=c_b):
    """Regime A: mandatory conventional issuance."""
    return mu*(RH-K-cA) + (1-mu)*(RL-K-cA)

def W_B(ds, RH=RH, RL=RL, cA=c_A, cb=c_b):
    """Regime B: opt-in STO, type H uses STO, type L uses conventional."""
    return mu*(RH-K-K*(1-ds)/ds-cb) + (1-mu)*(RL-K-cA)

def W_C(ds, RH=RH, RL=RL, cA=c_A, cb=c_b):
    """Regime C: mandatory complete non-issuance.
    H-type infeasible (P_C < K); only L-type survives with shortfall."""
    WL = RL - K - K*(1-ds)/ds - cb   # L-type surplus under mandatory STO
    return mu*0.0 + (1-mu)*WL

def delta_sigma_dom(cA=c_A, cb=c_b):
    return K / (K + cA - cb)

def in_sep_region(ds, RH=RH, RL=RL):
    return (K/RH) <= ds < (K/RL)

def welfare_ranking(ds, RH=RH, RL=RL, cA=c_A, cb=c_b):
    """Returns dict with welfare values and ranking."""
    wa = W_A(RH,RL,cA,cb)
    if not in_sep_region(ds,RH,RL):
        return {'W_A':wa, 'W_B':np.nan, 'W_C':np.nan,
                'sep_region':False, 'best':'A (sep. fails)'}
    wb = W_B(ds,RH,RL,cA,cb)
    wc = W_C(ds,RH,RL,cA,cb)
    dom = delta_sigma_dom(cA,cb)
    if ds >= dom:
        best = 'B ✓ (dominates)'
    else:
        best = 'A (dom. threshold not met)'
    return {'W_A':wa,'W_B':wb,'W_C':wc,
            'ΔW(B−A)':wb-wa,'P_C':mu*K+(1-mu)*ds*RL,
            'sep_region':True,'dom_threshold':dom,'best':best}

print("✓ Welfare functions defined.")

# ── CELL 4: Table OA.4 — δσ perturbation ─────────────────────────────────────
print("\n"+"━"*75)
print("  TABLE OA.4: WELFARE SENSITIVITY TO δσ PERTURBATION")
print("  Baseline: δσ = 0.720  (SMM estimate, Appendix B)")
print("━"*75)

DS_GRID = np.round(np.arange(0.55, 0.96, 0.05), 3)

print(f"\n  {'δσ':>6s}  {'In sep.':>8s}  {'W_A':>8s}  {'W_B':>8s}  "
      f"{'W_C':>8s}  {'ΔW(B−A)':>10s}  {'P_C':>8s}  {'Ranking':>25s}")
print("  "+"-"*90)

results_ds = []
for ds in DS_GRID:
    r = welfare_ranking(ds)
    results_ds.append({'ds':ds, **r})
    if r['sep_region']:
        print(f"  {ds:>6.3f}  {'✓':>8s}  {r['W_A']:>8.4f}  {r['W_B']:>8.4f}  "
              f"{r['W_C']:>8.4f}  {r['ΔW(B−A)']:>10.4f}  {r['P_C']:>8.4f}  "
              f"{r['best']:>25s}"
              + (" ← baseline" if abs(ds-ds_base)<0.01 else ""))
    else:
        print(f"  {ds:>6.3f}  {'✗':>8s}  {r['W_A']:>8.4f}  {'n/a':>8s}  "
              f"{'n/a':>8s}  {'n/a':>10s}  {'n/a':>8s}  {'sep. equil. fails':>25s}")

print(f"\n  Notes:")
print(f"  • δσ_dom = {K/(K+c_A-c_b):.4f}: threshold above which W_B > W_A")
print(f"  • Separation region: [{K/RH:.4f}, {K/RL:.4f})")
print(f"  • Baseline δσ = {ds_base:.4f} (marked ← baseline)")
print(f"  • W_B > W_C for all δσ in separation region (Proposition 5, part (ii))")
print(f"  • W_A > W_B at baseline: conventional issuance welfare-superior until")
print(f"    δσ improves by {K/(K+c_A-c_b)-ds_base:.3f} (+{(K/(K+c_A-c_b)/ds_base-1)*100:.1f}%)")

# ── CELL 5: Table OA.5 — Parameter sensitivity ───────────────────────────────
print("\n\n"+"━"*75)
print("  TABLE OA.5: SENSITIVITY OF δσ_dom TO c_A AND c_b")
print("  (Other parameters at SMM baseline)")
print("━"*75)

CA_GRID = [0.10, 0.15, 0.20, 0.219, 0.25, 0.30]
CB_GRID = [0.05, 0.10, 0.15, 0.181, 0.20]

print(f"\n  δσ_dom = K/(K + c_A − c_b):")
print(f"\n  {'':12s}", end="")
for cb in CB_GRID:
    print(f"  c_b={cb:.3f}", end="")
print()
print("  "+"-"*(12 + len(CB_GRID)*10))
for ca in CA_GRID:
    flag_ca = " ←SMM" if abs(ca-c_A)<0.001 else ""
    print(f"  c_A={ca:.3f}{flag_ca:5s}", end="")
    for cb in CB_GRID:
        if ca <= cb:
            print(f"  {'n/a':>8s}", end="")
        else:
            dom = K/(K+ca-cb)
            sep = in_sep_region(dom)
            flag = "✓" if sep else "✗"
            flag_cell = " ←SMM" if abs(ca-c_A)<0.001 and abs(cb-c_b)<0.001 else ""
            print(f"  {dom:.3f}{flag}{flag_cell:5s}", end="")
    print()

print(f"\n  ✓ = threshold inside separation region [{K/RH:.4f}, {K/RL:.4f})")
print(f"  ✗ = threshold outside separation region")
print(f"  SMM baseline: c_A={c_A:.3f}, c_b={c_b:.3f} → δσ_dom={K/(K+c_A-c_b):.4f} ✗")

# ── CELL 6: Sensitivity to RH and RL ─────────────────────────────────────────
print("\n\n"+"━"*75)
print("  TABLE OA.6: SENSITIVITY TO ASSET RETURNS R(H) AND R(L)")
print("━"*75)

RH_GRID = [1.30, 1.40, 1.548, 1.60, 1.80, 2.00]
RL_GRID = [1.05, 1.10, 1.140, 1.20, 1.30]

print(f"\n  Separation region upper bound K/R(L):")
print(f"\n  {'':12s}", end="")
for rl in RL_GRID:
    print(f"  RL={rl:.3f}", end="")
print()
for rh in RH_GRID:
    flag_rh = " ←SMM" if abs(rh-RH)<0.01 else ""
    print(f"  RH={rh:.3f}{flag_rh:5s}", end="")
    for rl in RL_GRID:
        if rh <= rl:
            print(f"  {'n/a':>8s}", end="")
        else:
            dsl = K/rh; dsh = K/rl
            dom = K/(K+c_A-c_b)
            in_sep = dsl <= ds_base < dsh
            in_dom = dsl <= dom < dsh
            flag_cell = " ←SMM" if abs(rh-RH)<0.01 and abs(rl-RL)<0.01 else ""
            status = f"{dsh:.3f}{'✓' if in_dom else '✗'}"
            print(f"  {status}{flag_cell:5s}", end="")
    print()

print(f"\n  Each cell shows K/R(L) (sep. region upper bound).")
print(f"  ✓/✗ = δσ_dom={K/(K+c_A-c_b):.3f} inside/outside separation region")

# ── CELL 7: Comprehensive welfare sensitivity surface ─────────────────────────
print("\n\n"+"━"*75)
print("  WELFARE SENSITIVITY: CONTINUOUS δσ AND c_A−c_b")
print("━"*75)

DS_CONT = np.linspace(K/RH+0.001, K/RL-0.001, 200)
DIFF_CONT = np.linspace(0.005, 0.10, 100)   # c_A - c_b grid

# For each (δσ, c_A-c_b), compute ΔW(B-A) = μ[(c_A-c_b) - K(1-δσ)/(δσ)]
DW_SURFACE = np.array([
    [mu*((diff) - K*(1-ds)/ds) for ds in DS_CONT]
    for diff in DIFF_CONT
])
DS_GRID_2D, DIFF_GRID_2D = np.meshgrid(DS_CONT, DIFF_CONT)

# ── CELL 8: Figures ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle('Welfare Sensitivity Analysis — Online Appendix OA.3\n'
             'Tables OA.4–OA.6',
             fontsize=13, fontweight='bold')

# (a) W_A, W_B, W_C vs δσ (Table OA.4 visualised)
ax = fig.add_subplot(gs[0,:2])
ds_plot = [r['ds'] for r in results_ds if r['sep_region']]
wa_plot = [r['W_A'] for r in results_ds if r['sep_region']]
wb_plot = [r['W_B'] for r in results_ds if r['sep_region']]
wc_plot = [r['W_C'] for r in results_ds if r['sep_region']]
ax.plot(ds_plot, wa_plot, 'o-', color='#E65100', lw=2.2, ms=7, label='$W_A$ (mandatory issuance)')
ax.plot(ds_plot, wb_plot, 's-', color='#1565C0', lw=2.2, ms=7, label='$W_B$ (opt-in STO)')
ax.plot(ds_plot, wc_plot, '^-', color='#6A1B9A', lw=2.0, ms=7, label='$W_C$ (mandatory STO)')
ax.axhline(0, color='black', lw=0.7)
ax.axvline(ds_base, color='grey', lw=1.2, ls=':', label=f'Baseline δσ={ds_base:.3f}')
ax.axvline(K/(K+c_A-c_b), color='#1565C0', lw=1.5, ls='--',
           label=f'δσ_dom={K/(K+c_A-c_b):.3f}')
ax.fill_between(ds_plot, wb_plot, wa_plot,
                where=[wb>wa for wb,wa in zip(wb_plot,wa_plot)],
                alpha=0.15, color='#1565C0', label='B dominates A')
ax.set_xlabel('Composite institutional quality δσ')
ax.set_ylabel('Aggregate welfare $W$')
ax.set_title('(a) Welfare by Regime vs δσ  (Table OA.4)')
ax.legend(fontsize=8.5)

# (b) ΔW(B-A) vs δσ
ax = fig.add_subplot(gs[0,2])
dw_plot = [r['ΔW(B−A)'] for r in results_ds if r['sep_region']]
ax.plot(ds_plot, dw_plot, 'o-', color='#1565C0', lw=2.2, ms=7)
ax.axhline(0, color='black', lw=1.0)
ax.fill_between(ds_plot, 0, dw_plot,
                where=[dw>0 for dw in dw_plot],
                alpha=0.25, color='#2E7D32', label='B > A')
ax.fill_between(ds_plot, dw_plot, 0,
                where=[dw<0 for dw in dw_plot],
                alpha=0.15, color='#C62828', label='A > B')
ax.axvline(ds_base, color='grey', lw=1.0, ls=':')
ax.axvline(K/(K+c_A-c_b), color='#1565C0', lw=1.5, ls='--')
ax.set_xlabel('δσ'); ax.set_ylabel('ΔW(B−A)')
ax.set_title('(b) ΔW(B−A) vs δσ\n'
             'Negative at baseline; reverses at δσ_dom')
ax.legend(fontsize=8.5)

# (c) ΔW surface: δσ × (c_A − c_b)
ax = fig.add_subplot(gs[1,:2])
cs = ax.contourf(DS_GRID_2D, DIFF_GRID_2D, DW_SURFACE,
                 levels=30, cmap='RdBu_r', vmin=-0.20, vmax=0.20)
ax.contour(DS_GRID_2D, DIFF_GRID_2D, DW_SURFACE, levels=[0],
           colors='black', linewidths=2.0)
plt.colorbar(cs, ax=ax, label='ΔW(B−A)')
ax.axvline(ds_base, color='white', lw=2.0, ls='--', label=f'Baseline δσ={ds_base:.3f}')
ax.axhline(c_A-c_b, color='white', lw=2.0, ls=':', label=f'Baseline c_A−c_b={c_A-c_b:.3f}')
ax.plot(ds_base, c_A-c_b, 'w*', ms=14, label='SMM estimate')
ax.set_xlabel('Composite institutional quality δσ')
ax.set_ylabel('Cost differential $c_A − c_b$')
ax.set_title('(c) ΔW(B−A) Surface  (Table OA.5 visualised)\n'
             'Black contour: ΔW = 0 (dominance boundary)')
ax.legend(fontsize=9, loc='upper left')

# (d) δσ_dom vs c_A−c_b with separation region
ax = fig.add_subplot(gs[1,2])
diff_arr = np.linspace(0.001, 0.35, 300)
dom_arr  = K/(K+diff_arr)
ax.plot(diff_arr, dom_arr, color='#1565C0', lw=2.5,
        label='δσ_dom = K/(K+c_A−c_b)')
ax.fill_between(diff_arr, K/RH, K/RL,
                alpha=0.15, color='#2E7D32', label=f'Sep. region [{K/RH:.3f},{K/RL:.3f})')
ax.fill_between(diff_arr, dom_arr, K/RL,
                where=(dom_arr>=K/RH)&(dom_arr<K/RL),
                alpha=0.25, color='#1565C0',
                label='Threshold inside sep. region → B dominates')
ax.axvline(c_A-c_b, color='grey', lw=1.2, ls=':',
           label=f'Baseline diff = {c_A-c_b:.3f}')
ax.axhline(ds_base, color='#C62828', lw=1.2, ls='--',
           label=f'Baseline δσ = {ds_base:.3f}')
ax.plot(c_A-c_b, K/(K+c_A-c_b), 'r*', ms=12, label='SMM estimate')
ax.set_xlabel('Cost differential $c_A - c_b$')
ax.set_ylabel('Dominance threshold δσ_dom')
ax.set_title('(d) Dominance Threshold vs Cost Differential\n'
             'SMM estimate: threshold outside sep. region')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0.5, 1.0)

# (e) ±0.05 perturbation bars (Associate Editor request)
ax = fig.add_subplot(gs[2,:])
perturb = np.array([-0.10, -0.05, 0.0, +0.05, +0.10])
ds_vals  = ds_base + perturb
wa_vals  = [W_A() for _ in perturb]
wb_vals  = [W_B(max(K/RH+0.001, min(K/RL-0.001, ds_base+p))) for p in perturb]
wc_vals  = [W_C(max(K/RH+0.001, min(K/RL-0.001, ds_base+p))) for p in perturb]
dw_vals  = [wb-wa for wb,wa in zip(wb_vals,wa_vals)]

x = np.arange(len(perturb)); w=0.22
ax.bar(x-w,     wa_vals, w, color='#E65100', alpha=0.85, label='$W_A$')
ax.bar(x,       wb_vals, w, color='#1565C0', alpha=0.85, label='$W_B$')
ax.bar(x+w,     wc_vals, w, color='#6A1B9A', alpha=0.85, label='$W_C$')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'δσ={v:.3f}\n(Δ={p:+.2f})' for v,p in zip(ds_vals,perturb)],
                   fontsize=9)
ax.set_ylabel('Aggregate welfare')
ax.set_title('(e) Welfare Ranking Under ±0.05 / ±0.10 Perturbation of δσ\n'
             '(Associate Editor request — Table OA.4)')
ax.legend(fontsize=9)
for i,(dw,ds) in enumerate(zip(dw_vals,ds_vals)):
    ax.text(i, max(wa_vals[i],wb_vals[i])+0.005,
            f'ΔW={dw:.3f}', ha='center', fontsize=8,
            color='#1565C0' if dw>0 else '#C62828')

plt.savefig('fig_sensitivity.pdf', bbox_inches='tight')
plt.show()
print("→ fig_sensitivity.pdf saved.")

# ── CELL 9: Compact sensitivity table for OA text ────────────────────────────
print("\n"+"━"*70)
print("  COMPACT TABLE OA.4 (for Online Appendix text)")
print("  Welfare values and ranking at δσ ± perturbation")
print("━"*70)

print(f"\n  {'Perturbation':>13s}  {'δσ':>6s}  {'W_A':>8s}  "
      f"{'W_B':>8s}  {'W_C':>8s}  {'ΔW(B−A)':>10s}  {'Ranking':>20s}")
print("  "+"-"*80)
for p in [-0.10, -0.05, 0.0, +0.05, +0.10]:
    ds_p = ds_base + p
    if not in_sep_region(ds_p): continue
    wa_p = W_A()
    wb_p = W_B(ds_p)
    wc_p = W_C(ds_p)
    dw_p = wb_p - wa_p
    rank = 'A > B > C' if wa_p>wb_p>wc_p else 'B > A > C'
    flag = '  ← baseline' if p==0.0 else ''
    print(f"  {p:>+13.2f}  {ds_p:>6.3f}  {wa_p:>8.4f}  "
          f"{wb_p:>8.4f}  {wc_p:>8.4f}  {dw_p:>+10.4f}  {rank:>20s}{flag}")

print(f"\n  Note: W_B > W_C for all δσ in separation region (constant across ϕ).")
print(f"  The ranking A > B reverses to B > A only when δσ ≥ {K/(K+c_A-c_b):.4f}.")
print(f"  At +0.10 perturbation (δσ = {ds_base+0.10:.3f}), ranking is still A > B.")
print(f"  Full reversal requires δσ ≥ {K/(K+c_A-c_b):.4f}: Δ = {K/(K+c_A-c_b)-ds_base:.3f} from baseline.")
