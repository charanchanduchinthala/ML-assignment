# Tutorial 09 — Gaussian Mixture Models & EM Algorithm
## Probabilistic Clustering — When K-Means Is Not Enough

**University of Hertfordshire | MLNN Assignment 2025**  
**Colour theme:** Burgundy (`#722F37`) / Sage Green (`#7FB069`)  
**GitHub:** https://github.com/yourusername/ml-tutorials/tree/main/tutorial-09

---

## Overview

This tutorial covers GMMs and the EM Algorithm from theory to real-data application:

- The GMM generative model: p(x) = Σₖ πₖ N(x|μₖ,Σₖ)
- E-step: soft responsibilities via Bayes' theorem
- M-step: weighted maximum-likelihood parameter updates
- EM convergence guarantee (monotone log-likelihood increase)
- GMM vs K-means on elongated, overlapping, and unequal-density clusters
- BIC model selection for choosing K — applied to real UCI Iris dataset
- sklearn `GaussianMixture` API with full covariance types

---

## Files

| File | Description |
|------|-------------|
| `tutorial_09_gmm_em.pdf` | Primary submission — PDF (<2000 words, 5 figures) |
| `tutorial_09_gmm_em.docx` | Word source document |
| `tutorial_09_gmm_em.ipynb` | Jupyter notebook — full runnable code, alt-text, references |
| `README.md` | This file |
| `LICENSE` | MIT Licence |
| `fig1_em_in_action.png` | 2×2 grid: random init → E-step → 5 iters → converged |
| `fig2_convergence_bic_iris.png` | Log-likelihood curve + BIC/AIC + Iris PCA scatter |
| `fig3_gmm_vs_kmeans.png` | Three cases where K-means fails, GMM succeeds |
| `fig4_estep_mstep.png` | Responsibility stacked bars + weighted scatter |
| `fig5_comparison.png` | Accuracy bar chart + covariance type scatter |

---

## How to Run

### Requirements

```bash
pip install numpy matplotlib scipy scikit-learn jupyter
```

### Launch

```bash
jupyter notebook tutorial_09_gmm_em.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`).  
All 5 figures regenerate and save as `.png` files.  
The UCI Iris dataset loads from `sklearn.datasets` — no download needed.

---

## Key Equations

**E-step** (responsibilities):
```
r_{nk} = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
```

**M-step** (parameter updates):
```
μ_k = Σ_n r_{nk} x_n / N_k
Σ_k = Σ_n r_{nk}(x_n-μ_k)(x_n-μ_k)ᵀ / N_k
π_k = N_k / N    where N_k = Σ_n r_{nk}
```

**BIC** (model selection):
```
BIC = -2 log L + p log N    (choose K that minimises BIC)
```

---

## Accessibility

- **Colourblind-safe palette** — burgundy (`#722F37`) and sage green (`#7FB069`) are distinguishable under deuteranopia, protanopia and tritanopia
- **Distinct marker shapes** on all scatter plots (circle `o`, square `s`, triangle `^`, diamond `D`, X marker for means)
- **Hatch patterns** on all bar charts (`//`, `xx`, `..`, `////`, `xxxx`) — information never by colour alone
- **Alt-text captions** printed below every figure cell in the notebook
- **Structured H1 → H2 heading hierarchy** for screen-reader navigation
- **High-contrast** dark (`#1A0F0F`) on light (`#FAFAF8`) — contrast ratio >14:1

---

## References

1. Dempster, A.P., Laird, N.M. and Rubin, D.B. (1977) 'Maximum likelihood from incomplete data via the EM algorithm', JRSS-B, 39(1). https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
2. Bishop, C.M. (2006) *Pattern Recognition and Machine Learning*. Springer. Chapter 9.
3. Reynolds, D.A. (2009) 'Gaussian mixture models', *Encyclopedia of Biometrics*. https://link.springer.com/referenceworkentry/10.1007/978-0-387-73003-5_196
4. McLachlan, G.J. and Peel, D. (2000) *Finite Mixture Models*. Wiley-Interscience.
5. Schwarz, G. (1978) 'Estimating the dimension of a model', *Annals of Statistics*, 6(2). https://doi.org/10.1214/aos/1176344136

---

**Licence:** MIT — see `LICENSE`
