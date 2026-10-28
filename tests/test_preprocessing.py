import numpy as np
import pandas as pd

from underwriting.preprocessing import build_preprocessor


def test_build_preprocessor():
    """Test numerical and categorical preprocessing."""

    # Create a small dataset with missing values.
    X = pd.DataFrame({
        "income": [1000.0, 2000.0, np.nan, 4000.0],
        "grade": ["A", "B", "A", np.nan],
    })

    # Construct and fit the preprocessing pipeline.
    preprocessor = build_preprocessor(
        numerical_features=["income"],
        categorical_features=["grade"],
    )

    X_transformed = preprocessor.fit_transform(X)
    print("Type:", type(X_transformed))
    print("Data type:", X_transformed.dtype)
    print("Shape:", X_transformed.shape)
    print("Values:", X_transformed)

    # Verify the expected output dimensions.
    assert X_transformed.shape == (4, 3)

    # Verify that no missing values remain.
    assert not np.isnan(X_transformed.data).any()

    # Verify the numerical imputation.
    numerical_imputer = (
        preprocessor.named_transformers_["num"]
        .named_steps["imputer"]
    )

    assert numerical_imputer.statistics_[0] == 2000.0

    # Verify the categorical imputation.
    categorical_imputer = (
        preprocessor.named_transformers_["cat"]
        .named_steps["imputer"]
    )

    assert categorical_imputer.statistics_[0] == "A"