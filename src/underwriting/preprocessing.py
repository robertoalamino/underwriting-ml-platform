from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numerical_features, categorical_features):
    """Build an unfitted numerical and categorical preprocessing pipeline."""
    
    # Construct an unfitted preprocessing pipeline.

    # Numerical features are median-imputed and standardised.
    # Categorical features are mode-imputed and one-hot encoded.

    # Parameters
    # ----------
    # numerical_features : list of str
    #     Names of numerical input features.
    # categorical_features : list of str
    #     Names of categorical input features.

    # Returns
    # -------
    # ColumnTransformer
    #     An unfitted scikit-learn preprocessing transformer.
    
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor