import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from underwriting.preprocessing import build_preprocessor


def test_model_persistence(tmp_path):
    """Test that saving and loading a fitted pipeline preserves predictions."""

    # Create a small synthetic classification dataset
    X = pd.DataFrame({
        "income": [1000.0, 2000.0, np.nan, 4000.0, 5000.0, 6000.0],
        "grade": ["A", "B", "A", "B", "A", "B"],
    })

    y = np.array([0, 0, 0, 1, 1, 1])

    # Construct the complete preprocessing and classification pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(
            numerical_features=["income"],
            categorical_features=["grade"],
        )),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # Fit the pipeline and generate reference predictions
    pipeline.fit(X, y)
    p_original = pipeline.predict_proba(X)[:, 1]

    # Save the fitted pipeline to a temporary file
    model_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    # Reload the pipeline and compare predictions
    loaded_pipeline = joblib.load(model_path)
    p_loaded = loaded_pipeline.predict_proba(X)[:, 1]

    np.testing.assert_array_equal(p_original, p_loaded)