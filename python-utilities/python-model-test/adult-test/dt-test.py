import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler
)
from dataclasses import dataclass


@dataclass
class PreprocessorBundle:
    standardizer: StandardScaler
    one_hot_encoder: OneHotEncoder
    label_encoder: LabelEncoder


def _remove_cols_and_nans(dataset: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "marital-status",
        "race",
        "occupation"
    ]

    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    return dataset.drop(columns_to_drop, axis=1)

def _fit_standardizer(X: pd.DataFrame) -> StandardScaler:
    columns_to_standardize = ["fnlwgt", "capitalgain", "capitalloss", "hoursperweek"]
    scaler = StandardScaler()
    scaler.fit(X[columns_to_standardize])
    return scaler

def _apply_standardizer(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    columns_to_standardize = ["fnlwgt", "capitalgain", "capitalloss", "hoursperweek"]
    X = X.copy()
    X[columns_to_standardize] = scaler.transform(X[columns_to_standardize])
    return X

def _fit_ohe(X: pd.DataFrame) -> OneHotEncoder:
    columns_to_encode = [
        "workclass",
        "education",
        "relationship",
        "native-country"
    ]
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    encoder.fit(X[columns_to_encode])
    return encoder

def _apply_ohe(X: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    columns_to_encode = [
        "workclass",
        "education",
        "relationship",
        "native-country"
    ]
    encoded = encoder.transform(X[columns_to_encode])
    X = X.drop(columns_to_encode, axis=1)
    return pd.concat([X, encoded], axis=1)

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
    ohe = _fit_ohe(X)
    le = _fit_label_encoder(Y)

    X = _apply_standardizer(X, scaler)
    X = _apply_ohe(X, ohe)
    Y = _apply_label_encoder(Y, le)

    bundle = PreprocessorBundle(scaler, ohe, le)
    return X, Y, bundle

def transform_preprocessors(X: pd.DataFrame, Y: pd.DataFrame,
                            bundle: PreprocessorBundle) -> tuple:
    dataset = pd.concat([X, Y], axis=1)
    dataset = _remove_cols_and_nans(dataset)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    X = _apply_standardizer(X, bundle.standardizer)
    X = _apply_ohe(X, bundle.one_hot_encoder)
    Y = _apply_label_encoder(Y, bundle.label_encoder)

    return X, Y

def main():
    dataset_paths = [
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/test-data/classification/adult/adult-train_x.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/test-data/classification/adult/adult-test_x.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/test-data/classification/adult/adult-train_y.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/test-data/classification/adult/adult-test_y.csv"
    ]
    
    train_x = pd.read_csv(dataset_paths[0])
    test_x = pd.read_csv(dataset_paths[1])
    train_y = pd.read_csv(dataset_paths[2])
    test_y = pd.read_csv(dataset_paths[3])

    X_train, Y_train, bundle = fit_preprocessors(train_x, train_y)
    X_test, Y_test = transform_preprocessors(test_x, test_y, bundle)
    
    X_train.to_csv(dataset_paths[0])
    Y_train.to_csv(dataset_paths[1])
    X_test.to_csv(dataset_paths[2])
    Y_test.to_csv(dataset_paths[3])
    
    dt_grid = {
        "criterion": "gini",
        "max_depth": 10,
        "max_feature": "sqrt",
        "max_leaf_nodes": 20,
        "min_samples_leaf": 20,
        "min_samples_split": 20,
        "min_impurity_decrease": 0.001
    }

    dt = DecisionTreeClassifier(**dt_grid)
    dt.fit(X_train, Y_train)

    predictions = dt.predict(X_test)
