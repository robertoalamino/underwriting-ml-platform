from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from underwriting.preprocessing import build_preprocessor


def train_logistic_regression(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
):
    """Fit a logistic regression model with integrated preprocessing."""

    # Construct the preprocessing and classification pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # Fit preprocessing and classification on the training data
    pipeline.fit(X_train, y_train)

    return pipeline