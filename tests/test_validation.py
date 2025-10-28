"""
Unit tests for validation module
Tests Purged K-Fold CV and probability calibration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.six_validation import (
    PurgedKFold, calibrate_probabilities, evaluate_calibration
)


class TestPurgedKFold(unittest.TestCase):
    """Test cases for Purged K-Fold cross-validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.purged_cv = PurgedKFold(
            n_splits=5,
            purge_pct=0.1,
            embargo_pct=0.05
        )

    def test_split_count(self):
        """Test that correct number of splits are generated"""
        X = np.random.randn(1000, 10)
        splits = self.purged_cv.split(X)

        self.assertEqual(len(splits), 5)

    def test_no_overlap_train_test(self):
        """Test that train and test sets don't overlap"""
        X = np.random.randn(1000, 10)
        splits = self.purged_cv.split(X)

        for train_idx, test_idx in splits:
            # Convert to sets
            train_set = set(train_idx)
            test_set = set(test_idx)

            # No overlap
            self.assertEqual(len(train_set.intersection(test_set)), 0)

    def test_purge_removes_samples(self):
        """Test that purging removes samples before test set"""
        X = np.random.randn(1000, 10)

        # Without purge
        cv_no_purge = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        splits_no_purge = cv_no_purge.split(X)

        # With purge
        cv_purge = PurgedKFold(n_splits=5, purge_pct=0.1, embargo_pct=0.0)
        splits_purge = cv_purge.split(X)

        # Purged version should have fewer training samples
        train_no_purge = splits_no_purge[0][0]
        train_purge = splits_purge[0][0]

        self.assertLess(len(train_purge), len(train_no_purge))

    def test_embargo_removes_samples(self):
        """Test that embargo removes samples after test set"""
        X = np.random.randn(1000, 10)

        # Without embargo
        cv_no_embargo = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        splits_no_embargo = cv_no_embargo.split(X)

        # With embargo
        cv_embargo = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.05)
        splits_embargo = cv_embargo.split(X)

        # Embargo version should have fewer training samples
        train_no_embargo = splits_no_embargo[0][0]
        train_embargo = splits_embargo[0][0]

        self.assertLess(len(train_embargo), len(train_no_embargo))

    def test_split_indices_valid(self):
        """Test that all split indices are valid"""
        X = np.random.randn(1000, 10)
        splits = self.purged_cv.split(X)

        for train_idx, test_idx in splits:
            # All indices should be within valid range
            self.assertTrue(np.all(train_idx >= 0))
            self.assertTrue(np.all(train_idx < len(X)))
            self.assertTrue(np.all(test_idx >= 0))
            self.assertTrue(np.all(test_idx < len(X)))


class TestProbabilityCalibration(unittest.TestCase):
    """Test cases for probability calibration"""

    def test_isotonic_calibration(self):
        """Test isotonic calibration"""
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        calibrated = calibrate_probabilities(y_true, y_prob, method="isotonic")

        # Calibrated probabilities should be in [0, 1]
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
        self.assertEqual(len(calibrated), len(y_prob))

    def test_platt_calibration(self):
        """Test Platt calibration"""
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        calibrated = calibrate_probabilities(y_true, y_prob, method="platt")

        # Calibrated probabilities should be in [0, 1]
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
        self.assertEqual(len(calibrated), len(y_prob))

    def test_calibration_with_perfect_predictions(self):
        """Test calibration with perfect predictions"""
        n_samples = 100
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_prob = y_true.copy()  # Perfect predictions

        calibrated = calibrate_probabilities(y_true, y_prob, method="isotonic")

        # Calibrated should be close to original
        np.testing.assert_array_almost_equal(calibrated, y_prob, decimal=1)

    def test_invalid_method_raises_error(self):
        """Test that invalid calibration method raises error"""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.8, 0.3, 0.9])

        with self.assertRaises(ValueError):
            calibrate_probabilities(y_true, y_prob, method="invalid")


class TestCalibrationEvaluation(unittest.TestCase):
    """Test cases for calibration evaluation"""

    def test_evaluate_calibration_returns_dataframe(self):
        """Test that evaluation returns DataFrame"""
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        result = evaluate_calibration(y_true, y_prob, n_bins=10)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("mean_predicted_prob", result.columns)
        self.assertIn("fraction_positive", result.columns)
        self.assertIn("calibration_error", result.columns)

    def test_calibration_error_non_negative(self):
        """Test that calibration error is non-negative"""
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        result = evaluate_calibration(y_true, y_prob, n_bins=10)

        # All calibration errors should be non-negative
        self.assertTrue(np.all(result["calibration_error"] >= 0))

    def test_perfect_calibration_zero_error(self):
        """Test that perfect calibration has near-zero error"""
        # Create perfectly calibrated probabilities
        n_samples = 1000
        y_prob = np.random.rand(n_samples)
        y_true = (np.random.rand(n_samples) < y_prob).astype(int)

        result = evaluate_calibration(y_true, y_prob, n_bins=5)

        # Average error should be small
        avg_error = result["calibration_error"].mean()
        self.assertLess(avg_error, 0.2)

    def test_different_bin_sizes(self):
        """Test evaluation with different bin sizes"""
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        result_5 = evaluate_calibration(y_true, y_prob, n_bins=5)
        result_10 = evaluate_calibration(y_true, y_prob, n_bins=10)

        # More bins should give more granular results
        self.assertGreaterEqual(len(result_10), len(result_5))


class TestCrossValidation(unittest.TestCase):
    """Integration tests for cross-validation workflow"""

    def test_cv_with_simple_model(self):
        """Test CV with a simple sklearn model"""
        from sklearn.linear_model import LogisticRegression
        from src.six_validation import cross_validate_model

        # Generate data
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Simple model
        model = LogisticRegression()

        # Run CV
        results = cross_validate_model(X, y, model, n_splits=3)

        self.assertIn("mean_accuracy", results)
        self.assertIn("std_accuracy", results)
        self.assertIn("fold_scores", results)

        # Accuracy should be reasonable
        self.assertGreater(results["mean_accuracy"], 0.5)
        self.assertLess(results["std_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
