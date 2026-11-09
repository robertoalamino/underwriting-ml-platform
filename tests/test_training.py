import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from underwriting.training import train_logistic_regression


def test_train_logistic_regression():
    """Test that the training function returns a fitted pipeline."""

    # Create a small synthetic classification dataset
    X = pd.DataFrame({
        "income": [1000.0, 2000.0, np.nan, 4000.0, 5000.0, 6000.0],
        "grade": ["A", "B", "A", "B", "A", "B"],
    })

    y = np.array([0, 0, 0, 1, 1, 1])

    # Train the complete model
    pipeline = train_logistic_regression(
        X_train=X,
        y_train=y,
        numerical_features=["income"],
        categorical_features=["grade"],
    )

    # Verify the returned object and its components
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # Verify that the fitted model generates valid probabilities
    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (6, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)