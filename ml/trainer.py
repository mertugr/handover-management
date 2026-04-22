"""
Random Forest trainer for the ML-based handover predictor.

Performs a user-level train/val/test split to avoid temporal leakage
(all samples from the same user stay in the same split), fits a
RandomForestClassifier, and caches the model with joblib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             classification_report)

from config import (RANDOM_SEED, RF_N_ESTIMATORS, RF_MAX_DEPTH,
                    RF_MIN_SAMPLES_LEAF, SPLIT_TEST_SIZE, SPLIT_VAL_SIZE)
from ml.features import FEATURE_COLS, LABEL_COL, build_feature_matrix
from simulation.cell_grid import NUM_CELLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")


@dataclass
class TrainReport:
    test_accuracy:   float = 0.0
    test_f1_macro:   float = 0.0
    val_accuracy:    float = 0.0
    train_accuracy:  float = 0.0
    n_train:         int   = 0
    n_val:           int   = 0
    n_test:          int   = 0
    feature_names:   list  = field(default_factory=list)
    feature_importances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    confusion:       np.ndarray     = field(default_factory=lambda: np.zeros(0))
    class_labels:    list           = field(default_factory=list)
    text_report:     str            = ""


def _split_by_user(df: pd.DataFrame, rng: np.random.RandomState
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """User-level train / val / test split to avoid leakage across time."""
    user_ids = np.array(sorted(df["user_id"].unique()))
    rng.shuffle(user_ids)

    n_users  = len(user_ids)
    n_test   = max(1, int(round(n_users * SPLIT_TEST_SIZE)))
    n_val    = max(1, int(round(n_users * SPLIT_VAL_SIZE)))
    n_train  = n_users - n_test - n_val
    if n_train <= 0:
        raise ValueError(f"Not enough users ({n_users}) for a train/val/test split")

    train_ids = set(user_ids[:n_train])
    val_ids   = set(user_ids[n_train:n_train + n_val])
    test_ids  = set(user_ids[n_train + n_val:])

    return (df[df["user_id"].isin(train_ids)],
            df[df["user_id"].isin(val_ids)],
            df[df["user_id"].isin(test_ids)])


def train(df: pd.DataFrame, save: bool = True) -> tuple[RandomForestClassifier, TrainReport]:
    """Train a RandomForestClassifier on the mobility dataset."""
    rng = np.random.RandomState(RANDOM_SEED)
    train_df, val_df, test_df = _split_by_user(df, rng)

    X_train, y_train = build_feature_matrix(train_df)
    X_val,   y_val   = build_feature_matrix(val_df)
    X_test,  y_test  = build_feature_matrix(test_df)

    clf = RandomForestClassifier(
        n_estimators       = RF_N_ESTIMATORS,
        max_depth          = RF_MAX_DEPTH,
        min_samples_leaf   = RF_MIN_SAMPLES_LEAF,
        n_jobs             = -1,
        random_state       = RANDOM_SEED,
        class_weight       = "balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred_test  = clf.predict(X_test)
    y_pred_val   = clf.predict(X_val)
    y_pred_train = clf.predict(X_train)

    labels = sorted(set(y_train) | set(y_val) | set(y_test))

    report = TrainReport(
        test_accuracy       = float(accuracy_score(y_test,  y_pred_test)),
        test_f1_macro       = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        val_accuracy        = float(accuracy_score(y_val,   y_pred_val)),
        train_accuracy      = float(accuracy_score(y_train, y_pred_train)),
        n_train             = int(len(y_train)),
        n_val               = int(len(y_val)),
        n_test              = int(len(y_test)),
        feature_names       = list(FEATURE_COLS),
        feature_importances = clf.feature_importances_.copy(),
        confusion           = confusion_matrix(y_test, y_pred_test,
                                               labels=list(range(NUM_CELLS))),
        class_labels        = list(range(NUM_CELLS)),
        text_report         = classification_report(y_test, y_pred_test,
                                                    labels=labels, zero_division=0),
    )

    if save:
        joblib.dump(clf, MODEL_PATH)
        print(f"[ML] Model saved -> {MODEL_PATH}")

    return clf, report


def load_model() -> Optional[RandomForestClassifier]:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def _evaluate_cached(model: RandomForestClassifier,
                     df: pd.DataFrame) -> TrainReport:
    """Re-evaluate a cached model on the same user-level split used for training."""
    rng = np.random.RandomState(RANDOM_SEED)
    train_df, val_df, test_df = _split_by_user(df, rng)

    X_train, y_train = build_feature_matrix(train_df)
    X_val,   y_val   = build_feature_matrix(val_df)
    X_test,  y_test  = build_feature_matrix(test_df)

    y_pred_test  = model.predict(X_test)
    y_pred_val   = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    labels = sorted(set(y_train) | set(y_val) | set(y_test))

    return TrainReport(
        test_accuracy       = float(accuracy_score(y_test,  y_pred_test)),
        test_f1_macro       = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        val_accuracy        = float(accuracy_score(y_val,   y_pred_val)),
        train_accuracy      = float(accuracy_score(y_train, y_pred_train)),
        n_train             = int(len(y_train)),
        n_val               = int(len(y_val)),
        n_test              = int(len(y_test)),
        feature_names       = list(FEATURE_COLS),
        feature_importances = model.feature_importances_.copy(),
        confusion           = confusion_matrix(y_test, y_pred_test,
                                               labels=list(range(NUM_CELLS))),
        class_labels        = list(range(NUM_CELLS)),
        text_report         = classification_report(y_test, y_pred_test,
                                                    labels=labels, zero_division=0),
    )


def train_or_load(df: pd.DataFrame, force_retrain: bool = False
                  ) -> tuple[RandomForestClassifier, TrainReport]:
    """Load a cached model when available, otherwise train a fresh one.

    Always returns a TrainReport so downstream code (plots, summary prints)
    does not need to re-train just to see the evaluation metrics.
    """
    if not force_retrain:
        model = load_model()
        if model is not None:
            # Feature layout may have shifted since the cache was written; if
            # so, discard the stale model and retrain rather than crashing.
            expected_n = len(FEATURE_COLS)
            cached_n   = getattr(model, "n_features_in_", None)
            if cached_n == expected_n:
                return model, _evaluate_cached(model, df)
            print(f"[ML] Cached model expects {cached_n} features, dataset "
                  f"provides {expected_n}. Retraining.")
    return train(df, save=True)
