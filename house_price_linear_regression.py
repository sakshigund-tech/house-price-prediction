"""
House Price Linear Regression
==============================
A from-scratch implementation of Ordinary Least Squares (OLS) linear regression
to predict house prices based on square footage, bedrooms, and bathrooms.

No ML libraries required — only Python standard library + optional numpy/matplotlib.

Usage:
    python house_price_linear_regression.py
"""

import math


# ---------------------------------------------------------------------------
# Dataset — 20 houses: (sqft, bedrooms, bathrooms, price)
# ---------------------------------------------------------------------------
DATASET = [
    (850,  2, 1, 145000),
    (1100, 2, 1, 182000),
    (1250, 3, 1, 198000),
    (1400, 3, 2, 221000),
    (1550, 3, 2, 245000),
    (1650, 3, 2, 262000),
    (1800, 4, 2, 278000),
    (1950, 4, 2, 295000),
    (2100, 4, 3, 315000),
    (2250, 4, 3, 334000),
    (2400, 5, 3, 358000),
    (2550, 4, 3, 372000),
    (2700, 5, 3, 391000),
    (2900, 5, 4, 418000),
    (3100, 5, 4, 445000),
    (3300, 6, 4, 469000),
    (3500, 6, 4, 492000),
    (3700, 6, 5, 521000),
    (4000, 7, 5, 558000),
    (4400, 7, 6, 612000),
]


# ---------------------------------------------------------------------------
# Matrix utilities (pure Python — no numpy needed)
# ---------------------------------------------------------------------------

def mat_transpose(M):
    """Transpose a 2-D list."""
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def mat_mul(A, B):
    """Multiply two 2-D lists."""
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result


def mat_inv(M):
    """
    Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
    Returns the inverse or raises ValueError if the matrix is singular.
    """
    n = len(M)
    # Augment [M | I]
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(M)]

    for col in range(n):
        # Partial pivot
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Matrix is singular — cannot invert.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        # Normalise pivot row
        factor = aug[col][col]
        aug[col] = [v / factor for v in aug[col]]

        # Eliminate column entries
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(2 * n)]

    return [row[n:] for row in aug]


def dot(a, b):
    """Dot product of two flat lists."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# OLS Linear Regression
# ---------------------------------------------------------------------------

class LinearRegression:
    """
    Ordinary Least Squares regression solved via the normal equation:

        β = (XᵀX)⁻¹ Xᵀy

    Supports multiple features (design matrix includes a bias column of 1s).
    """

    def __init__(self):
        self.coefficients = []   # [intercept, β1, β2, ...]
        self.feature_names = []

    def fit(self, X_cols, y, feature_names=None):
        """
        Train the model.

        Parameters
        ----------
        X_cols : list of lists — each inner list is one feature column
        y      : list of target values
        feature_names : optional list of strings
        """
        n = len(y)
        p = len(X_cols)

        # Build design matrix: column of 1s + feature columns
        X = [[1.0] + [X_cols[f][i] for f in range(p)] for i in range(n)]

        Xt   = mat_transpose(X)
        XtX  = mat_mul(Xt, X)
        XtX_inv = mat_inv(XtX)
        XtY  = [dot(row, y) for row in Xt]

        # β = (XᵀX)⁻¹ Xᵀy
        self.coefficients = [dot(row, XtY) for row in XtX_inv]
        self.feature_names = feature_names or [f"x{i}" for i in range(p)]
        return self

    def predict(self, *feature_values):
        """
        Predict for a single observation.

        Parameters
        ----------
        *feature_values : one argument per feature, in the same order as fit()

        Returns
        -------
        float : predicted target value
        """
        if not self.coefficients:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        intercept, *betas = self.coefficients
        return intercept + sum(b * x for b, x in zip(betas, feature_values))

    def predict_batch(self, X_cols):
        """Predict for a batch of observations (list of feature columns)."""
        n = len(X_cols[0])
        return [self.predict(*[X_cols[f][i] for f in range(len(X_cols))]) for i in range(n)]

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def r_squared(self, y_true, y_pred):
        """Coefficient of determination R²."""
        mean_y = mean(y_true)
        ss_tot = sum((y - mean_y) ** 2 for y in y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        return 1.0 - ss_res / ss_tot

    def rmse(self, y_true, y_pred):
        """Root Mean Squared Error."""
        n = len(y_true)
        return math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n)

    def mae(self, y_true, y_pred):
        """Mean Absolute Error."""
        n = len(y_true)
        return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, X_cols):
        """
        Scaled importance: |βᵢ × std(xᵢ)|
        Allows comparing features measured in different units.
        """
        _, *betas = self.coefficients
        return {
            name: abs(b * std(col))
            for name, b, col in zip(self.feature_names, betas, X_cols)
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, y_true, y_pred, X_cols):
        intercept, *betas = self.coefficients
        print("=" * 55)
        print("         Linear Regression — Model Summary")
        print("=" * 55)

        print("\n  Regression Equation:")
        eq = f"  Price = {intercept:,.0f}"
        for name, b in zip(self.feature_names, betas):
            sign = "+" if b >= 0 else "-"
            eq += f" {sign} {abs(b):,.2f} × {name}"
        print(eq)

        print("\n  Performance Metrics:")
        print(f"    R²   : {self.r_squared(y_true, y_pred):.6f}")
        print(f"    RMSE : ${self.rmse(y_true, y_pred):,.2f}")
        print(f"    MAE  : ${self.mae(y_true, y_pred):,.2f}")

        print("\n  Coefficients:")
        print(f"    {'Feature':<15} {'Coefficient':>14}")
        print(f"    {'-'*15} {'-'*14}")
        print(f"    {'Intercept':<15} {intercept:>14,.2f}")
        for name, b in zip(self.feature_names, betas):
            print(f"    {name:<15} {b:>14,.2f}")

        print("\n  Feature Importance (|β × std(x)|):")
        imp = self.feature_importance(X_cols)
        for name, score in sorted(imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(score / max(imp.values()) * 20)
            print(f"    {name:<15} {bar:<20} ${score:,.0f}")

        print("\n  Predictions vs Actuals (first 5 rows):")
        print(f"    {'#':<4} {'Actual':>10} {'Predicted':>12} {'Error':>10}")
        print(f"    {'-'*4} {'-'*10} {'-'*12} {'-'*10}")
        for i in range(5):
            err = y_true[i] - y_pred[i]
            sign = "+" if err >= 0 else ""
            print(f"    {i+1:<4} ${y_true[i]:>9,} ${y_pred[i]:>11,.0f} {sign}${err:>8,.0f}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Main — train, evaluate, and predict
# ---------------------------------------------------------------------------

def main():
    # Unpack dataset
    sqfts  = [row[0] for row in DATASET]
    beds   = [row[1] for row in DATASET]
    baths  = [row[2] for row in DATASET]
    prices = [row[3] for row in DATASET]

    # Train
    model = LinearRegression()
    model.fit(
        X_cols=[sqfts, beds, baths],
        y=prices,
        feature_names=["sqft", "bedrooms", "bathrooms"],
    )

    # Evaluate
    predictions = model.predict_batch([sqfts, beds, baths])
    model.summary(prices, predictions, [sqfts, beds, baths])

    # Interactive prediction
    print("\n  --- Predict a New House ---")
    examples = [
        (1800, 3, 2),
        (2500, 4, 3),
        (3200, 5, 4),
    ]
    print(f"\n  {'Sqft':<6} {'Beds':<6} {'Baths':<7} {'Estimated Price':>16}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*16}")
    for sqft, bed, bath in examples:
        price = model.predict(sqft, bed, bath)
        print(f"  {sqft:<6} {bed:<6} {bath:<7} ${price:>15,.0f}")

    print("\n  To predict your own house:")
    print("    model.predict(sqft=2200, bedrooms=4, bathrooms=3)")
    print(f"    → ${model.predict(2200, 4, 3):,.0f}\n")


if __name__ == "__main__":
    main()
