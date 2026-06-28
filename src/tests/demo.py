from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import (
    Lasso as SkLasso,
    LinearRegression as SkLinear,
    Ridge as SkRidge,
)
from sklearn.preprocessing import StandardScaler

from linear_regression_from_scratch import (
    LassoRegression,
    LinearRegression,
    RidgeRegression,
)

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[1]

ROOT = _repo_root()
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

MINE = "#0000ff"      # from-scratch
SK = "#ff0000"        # scikit-learn
DATA = "#64748b"
GRID = "#e2e8f0"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#94a3b8",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
    }
)


# metric helpers
def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))

# OLS on a simple 1-feature problem (interpretable line fit)
def section_ols() -> dict:
    print(f"1) Ordinary Least Squares  (mine vs scikit-learn)\n")
    rng = np.random.default_rng(0)
    n = 140
    X = rng.uniform(0, 10, size=(n, 1))
    y = (3.0 * X[:, 0] + 5.0) + rng.normal(0, 1.2, size=n)  # y = 3x + 5 + noise

    mine = LinearRegression(solver="normal").fit(X, y)
    sk = SkLinear().fit(X, y)

    rec = _compare("OLS", mine, sk, X, y)

    # chart: data + both fitted lines
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X[:, 0], y, s=22, color=DATA, alpha=0.6, label="data", zorder=2)
    xs = np.linspace(X.min(), X.max(), 100)
    ax.plot(xs, mine.w_[0] * xs + mine.b_, color=MINE, lw=2.6,
            label=f"mine  (y_hat = {mine.w_[0]:.2f}*x + {mine.b_:.2f})", zorder=3)
    ax.plot(xs, sk.coef_[0] * xs + sk.intercept_, color=SK, lw=2.2, ls="--",
            label=f"scikit-learn  (y_hat = {sk.coef_[0]:.2f}*x + {sk.intercept_:.2f})", zorder=3)
    ax.set_title("OLS fit - from-scratch vs scikit-learn")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper left")
    _save(fig, "01_ols_fit_comparison")

    # chart: residuals
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax_i, (name, pred, color) in zip(
        axes,
        [("mine", mine.predict(X), MINE), ("scikit-learn", sk.predict(X), SK)],
    ):
        ax_i.scatter(X[:, 0], y - pred, s=16, color=color, alpha=0.7)
        ax_i.axhline(0, color="#334155", lw=1)
        ax_i.set_title(f"{name} residuals"); ax_i.set_xlabel("x"); ax_i.set_ylabel("y - y_hat")
    fig.suptitle("OLS residuals - both centred on zero", fontweight="bold")
    _save(fig, "01_ols_residuals")

    _metrics_table(rec, "01_ols_metrics")
    return rec


# Ridge (L2) on multi-feature data
def section_ridge() -> dict:
    print("2) Ridge Regression (L2)\n")
    X, y, names = _multi_dataset(seed=1, n=300, d=8, informative=3, noise=0.8)
    Xs = StandardScaler().fit_transform(X)          # fair: regularise same space

    alpha_sk = 5.0
    mine = RidgeRegression(alpha=alpha_sk / X.shape[0], lr=0.5,
                           n_iter=5000, standardize=False).fit(Xs, y)
    sk = SkRidge(alpha=alpha_sk).fit(Xs, y)

    rec = _compare("Ridge (L2)", mine, sk, Xs, y)
    _coef_chart(mine.w_, sk.coef_, names, "Ridge coefficients - mine vs scikit-learn",
                "02_ridge_coefficients")
    _metrics_table(rec, "02_ridge_metrics")
    return rec


# Lasso (L1) - sparsity / feature selection
def section_lasso() -> dict:
    print("3) Lasso Regression (L1) - feature selection\n")
    X, y, names = _multi_dataset(seed=2, n=300, d=8, informative=3, noise=0.6)
    Xs = StandardScaler().fit_transform(X)

    alpha_sk = 0.5
    mine = LassoRegression(alpha=alpha_sk, lr=0.1, n_iter=30000,
                           standardize=False).fit(Xs, y)
    sk = SkLasso(alpha=alpha_sk, max_iter=50000).fit(Xs, y)

    rec = _compare("Lasso (L1)", mine, sk, Xs, y)
    _coef_chart(mine.w_, sk.coef_, names, "Lasso coefficients - sparsity matches",
                "03_lasso_coefficients")

    sparsity = pd.DataFrame(
        {
            "feature": names,
            "mine": np.round(mine.w_, 4),
            "scikit-learn": np.round(sk.coef_, 4),
            "mine==0": np.abs(mine.w_) < 1e-4,
            "sk==0": np.abs(sk.coef_) < 1e-4,
        }
    )
    _table_figure(sparsity, "Lasso sparsity (True = coefficient driven to zero)",
                  "03_lasso_sparsity")
    _metrics_table(rec, "03_lasso_metrics")
    return rec


# Prediction agreement across all models
def section_prediction_agreement(records: list[dict]) -> None:
    print("4) Prediction agreement (y_hat_mine vs y_hat_scikit-learn\n)")
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4.4))
    if len(records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        ax.scatter(rec["pred_mine"], rec["pred_sk"], s=14, color=MINE, alpha=0.6)
        lo = min(rec["pred_mine"].min(), rec["pred_sk"].min())
        hi = max(rec["pred_mine"].max(), rec["pred_sk"].max())
        ax.plot([lo, hi], [lo, hi], color=SK, ls="--", lw=1.5, label="y_hat_mine = y_hat_sk")
        ax.set_title(rec["name"]); ax.set_xlabel("y_hat  mine"); ax.set_ylabel("y_hat  scikit-learn")
        ax.legend(loc="upper left")
    fig.suptitle("Predictions agree to the diagonal", fontweight="bold")
    _save(fig, "04_prediction_agreement")


# Summary dashboard + combined CSV
def section_summary(records: list[dict]) -> None:
    print("5) Summary\n")

    # Combined CSV: one row per (model, metric)
    rows = []
    for rec in records:
        for metric in ["R2", "RMSE", "MAE"]:
            rows.append(
                {
                    "model": rec["name"],
                    "metric": metric,
                    "mine": rec[f"mine_{metric}"],
                    "scikit-learn": rec[f"sk_{metric}"],
                    "|diff|": abs(rec[f"mine_{metric}"] - rec[f"sk_{metric}"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(ASSETS / "comparison_summary.csv", index=False)

    _table_figure(
        summary.round(6),
        "Headline metrics - mine vs scikit-learn (|diff| ~= 0 means agreement)",
        "05_metrics_summary",
    )
    print(summary.round(4).to_string(index=False))
    print(f"\n wrote all comparisons to: {ASSETS}")


# shared helpers
def _multi_dataset(seed: int, n: int, d: int, informative: int, noise: float):
    """Deterministic multi-feature set; only informative columns carry signal"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d))
    true_w = np.zeros(d)
    true_w[:informative] = rng.uniform(2, 5, size=informative) * rng.choice([-1, 1], informative)
    y = X @ true_w + rng.normal(0, noise, size=n)
    names = [f"x{i + 1}{'  (signal)' if i < informative else '  (noise)'}" for i in range(d)]
    return X, y, names


def _compare(name: str, mine, sk, X: np.ndarray, y: np.ndarray) -> dict:
    """Fit-evaluation record comparing one pair of models"""
    pred_mine = np.asarray(mine.predict(X)).ravel()
    pred_sk = np.asarray(sk.predict(X)).ravel()
    rec = {
        "name": name,
        "mine_R2": r2(y, pred_mine), "sk_R2": r2(y, pred_sk),
        "mine_RMSE": rmse(y, pred_mine), "sk_RMSE": rmse(y, pred_sk),
        "mine_MAE": mae(y, pred_mine), "sk_MAE": mae(y, pred_sk),
        "coef_max_diff": float(np.max(np.abs(np.asarray(mine.w_).ravel() - np.asarray(sk.coef_).ravel()))),
        "pred_max_diff": float(np.max(np.abs(pred_mine - pred_sk))),
        "pred_mine": pred_mine, "pred_sk": pred_sk,
    }
    print(
        f"  R^2      mine={rec['mine_R2']:.5f}  sk={rec['sk_R2']:.5f}  "
        f"delta={abs(rec['mine_R2'] - rec['sk_R2']):.2e}\n"
        f"  RMSE    mine={rec['mine_RMSE']:.5f}  sk={rec['sk_RMSE']:.5f}\n"
        f"  max|delta_coef|={rec['coef_max_diff']:.2e}   max|delta_y_hat|={rec['pred_max_diff']:.2e}"
    )
    return rec


def _coef_chart(mine_w, sk_w, names, title, stem):
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, np.asarray(mine_w).ravel(), w, label="mine (from-scratch)", color=MINE)
    ax.bar(x + w / 2, np.asarray(sk_w).ravel(), w, label="scikit-learn", color=SK)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("  (signal)", "\n(signal)").replace("  (noise)", "\n(noise)")
                        for n in names], fontsize=8)
    ax.set_ylabel("coefficient (standardised space)")
    ax.set_title(title); ax.legend()
    _save(fig, stem)


def _metrics_table(rec: dict, stem):
    df = pd.DataFrame(
        {
            "metric": ["R^2", "RMSE", "MAE", "max |delta_coef|", "max |delta_y_hat|"],
            "mine (from-scratch)": [
                rec["mine_R2"], rec["mine_RMSE"], rec["mine_MAE"],
                rec["coef_max_diff"], rec["pred_max_diff"],
            ],
            "scikit-learn": [
                rec["sk_R2"], rec["sk_RMSE"], rec["sk_MAE"],
                rec["coef_max_diff"], rec["pred_max_diff"],
            ],
        }
    )
    df.to_csv(ASSETS / f"{stem}.csv", index=False)


def _table_figure(df: pd.DataFrame, title: str, stem: str):
    """Render a DataFrame as a clean labelled table image"""
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(df.columns)), 0.5 * len(df) + 1.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0e7490"); cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if r % 2 else "#eef2f7")
    ax.set_title(title, fontweight="bold", pad=12)
    _save(fig, stem)


def _save(fig, stem: str):
    path = ASSETS / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}")

def main() -> None:
    print(f"repo root : {ROOT}")
    print(f"assets dir: {ASSETS}")

    ols = section_ols()
    ridge = section_ridge()
    lasso = section_lasso()
    records = [ols, ridge, lasso]

    section_prediction_agreement(records)
    section_summary(records)

    print("Done")
    print("Generated files:")
    for p in sorted(ASSETS.iterdir()):
        print(f"{p.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
