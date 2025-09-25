"""Iris classification script with training, evaluation, CV, and CLI.

Provides utilities to load data, train a Logistic Regression model, perform
cross-validation, and save/load models. Designed for clarity and reproducible
experimentation.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib


@dataclass
class TrainConfig:
    """Training configuration for the Iris classifier.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Seed used by the random number generator for reproducibility.
    max_iter : int, default=200
        Maximum number of iterations taken for the solvers to converge.
    inverse_reg_strength : float, default=1.0
        Inverse of regularization strength; smaller values specify stronger
        regularization.
    penalty : {"l2", "l1", "elasticnet", "none"}, default="l2"
        Used to specify the norm used in the penalization (depends on solver).
    solver : {"lbfgs", "saga", "liblinear", "newton-cg", "sag"}, default="lbfgs"
        Algorithm to use in the optimization problem.
    multi_class : {"auto", "ovr", "multinomial"}, default="auto"
        If the option chosen is 'ovr', then a binary problem is fit for each
        label; otherwise, it is a multinomial loss.
    """
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 200
    inverse_reg_strength: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    multi_class: str = "auto"


def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load the Iris dataset.

    Returns
    -------
    x : ndarray of shape (n_samples, n_features)
        Feature matrix for Iris dataset (sepal length/width, petal length/width).
    y : ndarray of shape (n_samples,)
        Target labels encoded as integers {0, 1, 2}.
    target_names : list of str
        Human-readable class names in index order corresponding to `y` labels.

    Notes
    -----
    Data is loaded from scikit-learn's built-in datasets module.
    """
    x, y = load_iris(return_X_y=True)
    iris_bunch = load_iris()
    target_names = list(iris_bunch["target_names"])
    return x, y, target_names


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[LogisticRegression, dict]:
    """Train a Logistic Regression classifier on Iris and compute test metrics.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels encoded as integers.
    cfg : TrainConfig
        Training configuration including split and model hyperparameters.

    Returns
    -------
    model : LogisticRegression
        Fitted scikit-learn classifier.
    metrics : dict
        Dictionary with keys 'accuracy' (float), 'report' (dict), and
        'confusion_matrix' (list of lists).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.inverse_reg_strength,
        penalty=cfg.penalty,
        solver=cfg.solver,
        multi_class=cfg.multi_class,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def cross_validate(
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """Perform k-fold cross-validation and summarize accuracy.

    Parameters
    ----------
    model : LogisticRegression
        Unfitted estimator to be evaluated via cross-validation.
    x : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels.
    cv : int, default=5
        Number of folds for cross-validation.

    Returns
    -------
    summary : dict
        Contains 'cv' (int), 'mean_accuracy' (float), 'std_accuracy' (float),
        and 'all_scores' (list of float).
    """
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
    return {
        "cv": cv,
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "all_scores": scores.tolist(),
    }


def save_model(model: LogisticRegression, path: str) -> None:
    """Persist a trained model to disk using Joblib.

    Parameters
    ----------
    model : LogisticRegression
        Trained scikit-learn model to serialize.
    path : str
        Destination file path (e.g., 'iris_lr.joblib').
    """
    joblib.dump(model, path)


def load_model(path: str) -> LogisticRegression:
    """Load a serialized model from disk using Joblib.

    Parameters
    ----------
    path : str
        Path to a previously saved Joblib file.

    Returns
    -------
    model : LogisticRegression
        Deserialized scikit-learn model instance.
    """
    return joblib.load(path)


def main():
    """Run training/evaluation for the Iris classifier with CLI options.

    The script prints a JSON report including parameters, metrics, optional
    cross-validation summary, and optional predictions. It also supports
    saving the trained model to disk.

    Notes
    -----
    See `--help` for available CLI flags and their defaults.
    """
    parser = argparse.ArgumentParser(
        description="Iris classifier (Logistic Regression)"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations")
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (C)",
    )
    parser.add_argument("--penalty", type=str, default="l2", help="Penalty type")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Solver")
    parser.add_argument("--multi-class", type=str, default="auto", help="Multi-class strategy")
    parser.add_argument("--cv", type=int, default=0, help="k-fold CV; 0 to disable")
    parser.add_argument(
        "--save-model",
        type=str,
        default="",
        help="Path to save trained model (joblib)",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="",
        help=(
            "JSON array of samples to predict, e.g. "
            "'[[5.1, 3.5, 1.4, 0.2]]'"
        ),
    )

    args = parser.parse_args()
    cfg = TrainConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        inverse_reg_strength=args.C,
        penalty=args.penalty,
        solver=args.solver,
        multi_class=args.multi_class,
    )

    x, y, target_names = load_data()
    base_model = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.inverse_reg_strength,
        penalty=cfg.penalty,
        solver=cfg.solver,
        multi_class=cfg.multi_class,
    )

    cv_metrics = None
    if args.cv and args.cv > 0:
        cv_metrics = cross_validate(base_model, x, y, cv=args.cv)

    model, metrics = train_model(x, y, cfg)

    if args.save_model:
        save_model(model, args.save_model)

    predictions = None
    if args.predict:
        samples = json.loads(args.predict)
        samples = np.array(samples)
        preds = model.predict(samples).tolist()
        predictions = {
            "samples": samples.tolist(),
            "predictions": preds,
            "target_names": target_names,
            "predicted_labels": [target_names[i] for i in preds],
        }

    print(
        json.dumps(
            {
                "params": cfg.__dict__,
                "metrics": metrics,
                "cv_metrics": cv_metrics,
                "target_names": target_names,
                "saved_model": bool(args.save_model),
                "predictions": predictions,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
