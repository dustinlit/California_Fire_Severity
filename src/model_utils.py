# src/eval_utils.py

import numpy as np
import pandas as pd


def gen_report(reports):
    """
    Aggregate average precision, recall, and F1-score by class over k-folds.

    Parameters:
        reports (list of dicts): Each dict is a classification_report-style output
                                 with class-level precision, recall, f1-score.

    Returns:
        pd.DataFrame: Averaged metrics per class (Precision, Recall, F1-Score)
    """
    classes = ['0', '1', '2']
    metrics = ['precision', 'recall', 'f1-score']

    # Initialize structure: {class: {metric: []}}
    results = {cls: {m: [] for m in metrics} for cls in classes}

    # Collect values
    for report in reports:
        for cls in classes:
            for metric in metrics:
                results[cls][metric].append(report[cls][metric])

    # Compute means
    data = {
        'Category': classes,
        'Precision': [np.mean(results[cls]['precision']) for cls in classes],
        'Recall':    [np.mean(results[cls]['recall']) for cls in classes],
        'F1-Score':  [np.mean(results[cls]['f1-score']) for cls in classes]
    }

    return pd.DataFrame(data)

# src/model_utils.py

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def kfold(X_train, y_train, model):
    """
    Perform 5-fold cross-validation and return classification reports.

    Parameters:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable.
        model: An instantiated (but untrained) model.
        encoding (bool): If True, label encode y values (e.g. for XGBoost).

    Returns:
        List[dict]: A list of classification report dictionaries (one per fold).
    """
        
    report_dicts = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train and predict
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        report = classification_report(y_test_fold, y_pred, output_dict=True)
        report_dicts.append(report)

    return report_dicts

def kfold_old(X_train, y_train, model, encoding=False):
    """
    Perform 5-fold cross-validation and return classification reports.

    Parameters:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable.
        model: An instantiated (but untrained) model.
        encoding (bool): If True, label encode y values (e.g. for XGBoost).

    Returns:
        List[dict]: A list of classification report dictionaries (one per fold).
    """
    if encoding:
        mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        y_train = y_train.map(mapping).astype(int)
        
    report_dicts = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train and predict
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        # Re-encode predictions if needed
        if encoding:
            y_pred = pd.Series(y_pred).map(inverse_mapping)
            y_test_fold = y_test_fold.map(inverse_mapping)

        report = classification_report(y_test_fold, y_pred, output_dict=True)
        report_dicts.append(report)

    return report_dicts

