import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def split_columns(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    binary_cols = [col for col in num_cols if X[col].nunique() == 2]
    cont_cols   = [col for col in num_cols if X[col].nunique() > 2]

    return cat_cols, cont_cols, binary_cols

def build_preprocessor(cat_cols, cont_cols, binary_cols, requires_encoding):
    if requires_encoding:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
    else:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

    cont_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("cat",    cat_transformer,  cat_cols),
        ("cont",   cont_transformer, cont_cols),
        ("binary", "passthrough",    binary_cols),
    ], remainder="drop")
