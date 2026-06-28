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


## Part 1. What this project is (in plain English)

This section is written for **non-technical readers and developers who have not
worked with linear regression before**.

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
