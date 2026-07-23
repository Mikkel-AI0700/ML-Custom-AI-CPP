from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler
)
from dataclasses import dataclass

COLUMNS_TO_DROP = ["marital-status", "race", "occupation", "native-country"]
COLUMNS_TO_STANDARDIZE = ["fnlwgt", "capitalgain", "capitalloss", "hoursperweek", "age", "education-num"]
COLUMNS_TO_ENCODE = ["workclass", "relationship", "sex"]
COLUMNS_TO_ORDINAL = ["education"]
EDUCATION_CATEGORIES = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
    "11th", "12th", "HS-grad", "Some-college",
    "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
    "Prof-school", "Doctorate"
]


@dataclass
class PreprocessorBundle:
    standardizer: StandardScaler
    ordinal_encoder: OrdinalEncoder
    one_hot_encoder: OneHotEncoder
    label_encoder: LabelEncoder


def _remove_cols_and_nans(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    return dataset.drop(COLUMNS_TO_DROP, axis=1)

def _fit_standardizer(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X[COLUMNS_TO_STANDARDIZE])
    return scaler

def _apply_standardizer(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = X.copy()
    X[COLUMNS_TO_STANDARDIZE] = scaler.transform(X[COLUMNS_TO_STANDARDIZE])
    return X

def _fit_ohe(X: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
    encoder.fit(X[COLUMNS_TO_ENCODE])
    return encoder

def _apply_ohe(X: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    encoded = encoder.transform(X[COLUMNS_TO_ENCODE])
    X = X.drop(COLUMNS_TO_ENCODE, axis=1)
    result = pd.concat([X, encoded], axis=1)
    categorical_features = list(encoder.get_feature_names_out())
    result = result[COLUMNS_TO_STANDARDIZE + COLUMNS_TO_ORDINAL + categorical_features]
    return result

def _fit_ordinal(X: pd.DataFrame) -> OrdinalEncoder:
    encoder = OrdinalEncoder(
        categories=[EDUCATION_CATEGORIES],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ).set_output(transform="pandas")
    encoder.fit(X[COLUMNS_TO_ORDINAL])
    return encoder

def _apply_ordinal(X: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    X = X.copy()
    X[COLUMNS_TO_ORDINAL] = encoder.transform(X[COLUMNS_TO_ORDINAL])
    return X

def _fit_label_encoder(Y: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(Y)
    return encoder

def _apply_label_encoder(Y: pd.Series, encoder: LabelEncoder) -> pd.Series:
    return encoder.transform(Y)

def fit_preprocessors(X: pd.DataFrame, Y: pd.DataFrame) -> tuple:
    dataset = pd.concat([X, Y], axis=1)
    dataset = _remove_cols_and_nans(dataset)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    scaler = _fit_standardizer(X)
    ordinal = _fit_ordinal(X)
    ohe = _fit_ohe(X)
    le = _fit_label_encoder(Y)

    X = _apply_standardizer(X, scaler)
    X = _apply_ordinal(X, ordinal)
    X = _apply_ohe(X, ohe)
    Y = _apply_label_encoder(Y, le)

    bundle = PreprocessorBundle(scaler, ordinal, ohe, le)
    return X, Y, bundle

def transform_preprocessors(X: pd.DataFrame, Y: pd.DataFrame,
                            bundle: PreprocessorBundle) -> tuple:
    dataset = pd.concat([X, Y], axis=1)
    dataset = _remove_cols_and_nans(dataset)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    X = _apply_standardizer(X, bundle.standardizer)
    X = _apply_ordinal(X, bundle.ordinal_encoder)
    X = _apply_ohe(X, bundle.one_hot_encoder)
    Y = _apply_label_encoder(Y, bundle.label_encoder)
    return X, Y


def _write_features_file(dataset_dir: Path,
                         bundle: PreprocessorBundle) -> None:
    numerical = sorted(COLUMNS_TO_STANDARDIZE)
    ordinal = COLUMNS_TO_ORDINAL
    categorical = list(bundle.one_hot_encoder.get_feature_names_out())

    path = dataset_dir / "features.txt"
    with open(path, "w") as f:
        f.write(f"# Numerical features ({len(numerical)}):\n")
        for col in numerical:
            f.write(f"{col}\n")
        f.write("\n")
        f.write(f"# Ordinal features ({len(ordinal)}):\n")
        for col in ordinal:
            f.write(f"{col}\n")
        f.write("\n")
        f.write(f"# Categorical features ({len(categorical)}):\n")
        for col in categorical:
            f.write(f"{col}\n")


def main():
    BASE = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = BASE / "datasets" / "openml" / "adult"
    
    # Raw CSVs from get_dataset.py have no headers
    # Define original column names from the dataset
    raw_columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capitalgain", "capitalloss", "hoursperweek", "native-country"
    ]
    
    train_x = pd.read_csv(dataset_dir / "train_x.csv", header=None, names=raw_columns)
    test_x = pd.read_csv(dataset_dir / "test_x.csv", header=None, names=raw_columns)
    train_y = pd.read_csv(dataset_dir / "train_y.csv", header=None, names=["target"])
    test_y = pd.read_csv(dataset_dir / "test_y.csv", header=None, names=["target"])

    X_train, Y_train, bundle = fit_preprocessors(train_x, train_y)
    X_test, Y_test = transform_preprocessors(test_x, test_y, bundle)

    _write_features_file(dataset_dir, bundle)

    # Save as .npy (binary, no headers, exact shape/dtype)
    np.save(dataset_dir / "train_x.npy", X_train.to_numpy())
    np.save(dataset_dir / "train_y.npy", Y_train)
    np.save(dataset_dir / "test_x.npy", X_test.to_numpy())
    np.save(dataset_dir / "test_y.npy", Y_test)

    # Also save headerless CSV for C++ compatibility
    pd.DataFrame(X_train).to_csv(dataset_dir / "train_x.csv", index=False, header=False)
    pd.Series(Y_train, name="target").to_csv(dataset_dir / "train_y.csv", index=False, header=False)
    pd.DataFrame(X_test).to_csv(dataset_dir / "test_x.csv", index=False, header=False)
    pd.Series(Y_test, name="target").to_csv(dataset_dir / "test_y.csv", index=False, header=False)
    
    dt_grid = {
        "criterion": "gini",
        "max_depth": 10,
        #"max_features": "sqrt",
        "max_leaf_nodes": 10,
        "min_samples_leaf": 20,
        "min_samples_split": 20,
        "min_impurity_decrease": 0.0001
    }

    dt = DecisionTreeClassifier(**dt_grid)
    dt.fit(X_train, Y_train)

    predictions = dt.predict(X_test)

    preds_dir = dataset_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    # Save as .npy and headerless CSV
    np.save(preds_dir / "python-predictions.npy", predictions)
    pd.Series(predictions, name="target").to_csv(preds_dir / "python-predictions.csv", index=False, header=False)


if __name__ == "__main__":
    main()
