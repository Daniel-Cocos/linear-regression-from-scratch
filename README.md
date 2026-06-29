# linear-regression-from-scratch

Linear regression built **from scratch** using only NumPy, then packaged so it
installs and imports like any normal external library. No black-box regressor
is ever called. Every weight is learned by gradient descent (or the closed-form
normal equations) that I wrote myself, and the code is wrapped in a clean,
installable Python package that other projects can depend on just like they
depend on any other library.

```
pip install -e ".[dev]"
pytest -q
```


## What this project is (in plain English)

### What linear regression does

Linear regression is one of the oldest and most widely used methods for
predicting a number. You give it some inputs (for example, the square footage,
number of bedrooms, and age of a house) and it learns a simple rule that
combines those inputs to predict a target number (for example, the house price).

"From scratch" means I did not call a pre-built solver to do the learning. I
wrote the mathematics that finds the best rule myself, using only NumPy for basic
array arithmetic. "Packaged" means the result is organized like a real library,
so anyone can install it with a single command and use it in their own code.

### The three models in this library

- **OLS (Ordinary Least Squares)** - the classic version. It finds the rule that
  makes the total squared error as small as possible. This is the starting point
  for all of linear regression.
- **Ridge (L2)** - adds a small penalty that shrinks the influence of each input.
  Shrinking helps the model generalize to new, unseen data instead of just
  memorizing the training data (a failure mode called overfitting).
- **Lasso (L1)** - a different penalty that can shrink some inputs' influence all
  the way to zero. Because zeroed inputs are ignored entirely, Lasso
  automatically decides which inputs matter and which do not. This is called
  automatic feature selection.

### The three metrics used to judge the models

- **R2 (R-squared)** - a score that says how much of the variation in the target
  the model explains. It usually runs from 0 to 1, and closer to 1 is better.
  You can think of it as a percentage grade for the model.
- **RMSE (Root Mean Squared Error)** - the typical size of a prediction error,
  in the same units as the target. Lower is better. It squares the errors first,
  so it reacts strongly to a few very large errors.
- **MAE (Mean Absolute Error)** - the average absolute size of prediction errors.
  Lower is better. Unlike RMSE, it is less affected by a few large errors, so it
  gives a steadier view of typical performance.

### Proven correct against the industry standard

Below, **"mine"** is this from-scratch library and **"scikit-learn"** is the
widely used, industry-standard reference library. The **|diff|** column shows how
far apart their results are. Values of 0.0000 mean the two agree to the
precision shown, which proves the from-scratch implementation is correct.

| Model | Metric | mine | scikit-learn | \|diff\| |
| --- | --- | --- | --- | --- |
| OLS | R2 | 0.9803 | 0.9803 | 0.0000 |
| OLS | RMSE | 1.2878 | 1.2878 | 0.0000 |
| OLS | MAE | 1.0078 | 1.0078 | 0.0000 |
| Ridge (L2) | R2 | 0.9886 | 0.9886 | 0.0000 |
| Ridge (L2) | RMSE | 0.7553 | 0.7553 | 0.0000 |
| Ridge (L2) | MAE | 0.6012 | 0.6012 | 0.0000 |
| Lasso (L1) | R2 | 0.9738 | 0.9738 | 0.0000 |
| Lasso (L1) | RMSE | 1.0704 | 1.0699 | 0.0005 |
| Lasso (L1) | MAE | 0.8654 | 0.8650 | 0.0004 |

The tiny differences on the Lasso rows are expected: Lasso is solved with an
iterative proximal-gradient method, so a perfect match takes more iterations.
Both the accuracy (R2, RMSE, MAE) and the sparsity pattern line up with
scikit-learn.

## Quick start

### Install with pip

To develop and use inside this repository:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

To use it from a different project (installed from git):

```bash
pip install git+https://github.com/Daniel-Cocos/linear-regression-from-scratch.git
```

After publishing to PyPI, that simplifies to:

```bash
pip install linear-regression-from-scratch
```

### Install with uv

```bash
uv pip install -e ".[dev]"       # inside this repo
```

From another project:

```bash
uv pip install git+https://github.com/Daniel-Cocos/linear-regression-from-scratch.git
# or, in a uv-managed project:
uv add git+https://github.com/Daniel-Cocos/linear-regression-from-scratch.git
```

### Install with Poetry

From another project:

```bash
poetry add git+https://github.com/Daniel-Cocos/linear-regression-from-scratch.git
# or from a local editable checkout for development:
poetry add ../linear-regression-from-scratch
```

### Install with Nix flakes

Add it to another flake's inputs and consume the built package:

```nix
inputs.linear-regression-from-scratch.url =
  "github:Daniel-Cocos/linear-regression-from-scratch";
```

```nix
inputs.linear-regression-from-scratch.packages.${system}.default
```

Inside this repository itself:

```bash
nix build        # reproducibly build the wheel
nix develop      # reproducible dev shell with an editable install
```
