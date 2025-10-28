"""
Validation Module: Purged K-Fold CV, Embargo, and Probability Calibration
Prevents temporal leakage in time-series predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging
from typing import List, Tuple

from config import *

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time-series
    Removes training samples that overlap with test period
    """

    def __init__(
        self,
        n_splits: int = CV_CONFIG["n_splits"],
        purge_pct: float = CV_CONFIG["purge_pct"],
        embargo_pct: float = CV_CONFIG["embargo_pct"]
    ):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits

        Args:
            X: Feature array (only shape matters)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = X.shape[0]
        test_size = n_samples // self.n_splits
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)

        indices = np.arange(n_samples)
        splits = []

        for i in range(self.n_splits):
            # Test set
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            # Purge before test
            purge_start = max(0, test_start - purge_size)

            # Embargo after test
            embargo_end = min(n_samples, test_end + embargo_size)

            # Train set: all except purged, test, and embargoed
            train_indices = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"Generated {len(splits)} purged CV splits")
        return splits


def calibrate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = CALIBRATION_CONFIG["method"]
) -> np.ndarray:
    """
    Calibrate predicted probabilities

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        method: Calibration method ("isotonic" or "platt")

    Returns:
        Calibrated probabilities
    """
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrated = calibrator.fit_transform(y_prob, y_true)
    elif method == "platt":
        # For binary, use logistic regression on probabilities
        y_prob_reshaped = y_prob.reshape(-1, 1)
        calibrator = LogisticRegression()
        calibrator.fit(y_prob_reshaped, y_true)
        calibrated = calibrator.predict_proba(y_prob_reshaped)[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    logger.info(f"Probabilities calibrated using {method}")
    return calibrated


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = CALIBRATION_CONFIG["n_bins"]
) -> pd.DataFrame:
    """
    Evaluate calibration quality

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        DataFrame with calibration statistics
    """
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_prob": mean_predicted_value,
        "fraction_positive": fraction_of_positives,
        "calibration_error": np.abs(mean_predicted_value - fraction_of_positives)
    })

    expected_calibration_error = calibration_df["calibration_error"].mean()
    logger.info(f"Expected Calibration Error: {expected_calibration_error:.4f}")

    return calibration_df


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = CV_CONFIG["n_splits"]
) -> dict:
    """
    Perform purged cross-validation

    Args:
        X: Feature array
        y: Label array
        model: Model with fit/predict interface
        n_splits: Number of CV splits

    Returns:
        Dictionary with CV results
    """
    purged_cv = PurgedKFold(n_splits=n_splits)

    scores = []
    all_predictions = []
    all_true_labels = []

    for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
        logger.info(f"CV Fold {fold + 1}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        scores.append(accuracy)

        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)

        logger.info(f"Fold {fold + 1} accuracy: {accuracy:.4f}")

    results = {
        "mean_accuracy": np.mean(scores),
        "std_accuracy": np.std(scores),
        "fold_scores": scores,
        "predictions": np.array(all_predictions),
        "true_labels": np.array(all_true_labels)
    }

    logger.info(
        f"CV Results: Mean Accuracy = {results['mean_accuracy']:.4f} "
        f"+/- {results['std_accuracy']:.4f}"
    )

    return results


def main():
    """Main execution function"""
    logger.info("Validation module example")

    # Generate synthetic data
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Test Purged K-Fold
    logger.info("Testing Purged K-Fold")
    purged_cv = PurgedKFold(n_splits=5)
    splits = purged_cv.split(X)

    for i, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

    # Test calibration
    logger.info("\nTesting calibration")
    y_prob = np.random.rand(n_samples)
    calibrated = calibrate_probabilities(y, y_prob, method="isotonic")

    logger.info(f"Original prob range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    logger.info(f"Calibrated prob range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")

    # Evaluate calibration
    calibration_df = evaluate_calibration(y, y_prob, n_bins=10)
    logger.info(f"\nCalibration statistics:\n{calibration_df}")

    logger.info("Validation module complete")


if __name__ == "__main__":
    main()
