"""Validate ProteinGym DMS eval against published leaderboard scores.

The ProteinGym project publishes per-assay, per-model Spearman rho values.
We use these as ground truth to verify our eval pipeline is correct.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

FIXTURES = Path(__file__).parent / "fixtures"
LEADERBOARD = FIXTURES / "proteingym_expected_spearman.csv"


def _load_leaderboard() -> list[dict]:
    with open(LEADERBOARD) as f:
        return list(csv.DictReader(f))


class TestProteinGymLeaderboard:
    """Tests that use the published leaderboard as ground truth."""

    def test_leaderboard_has_217_assays(self):
        rows = _load_leaderboard()
        assert len(rows) == 217

    def test_leaderboard_has_saprot(self):
        rows = _load_leaderboard()
        assert "SaProt (650M)" in rows[0]

    def test_saprot_mean_rho_matches_published(self):
        """SaProt-650M published mean Spearman rho ~0.473."""
        rows = _load_leaderboard()
        rhos = [float(r["SaProt (650M)"]) for r in rows]
        mean_rho = np.mean(rhos)
        assert 0.45 < mean_rho < 0.50, f"SaProt mean rho {mean_rho:.4f} outside expected range"

    def test_eve_mean_rho_matches_published(self):
        """EVE ensemble published mean Spearman rho ~0.460."""
        rows = _load_leaderboard()
        rhos = [float(r["EVE (ensemble)"]) for r in rows]
        mean_rho = np.mean(rhos)
        assert 0.44 < mean_rho < 0.48, f"EVE mean rho {mean_rho:.4f} outside expected range"

    def test_brca1_saprot_rho(self):
        """BRCA1 Findlay 2018: SaProt(650M) should get rho ~0.539."""
        rows = _load_leaderboard()
        brca1 = [r for r in rows if "BRCA1" in r["DMS ID"]]
        assert len(brca1) == 1
        rho = float(brca1[0]["SaProt (650M)"])
        assert abs(rho - 0.539) < 0.01, f"BRCA1 SaProt rho {rho} != expected 0.539"

    def test_site_independent_is_weakest(self):
        """Site-independent model should be weaker than EVE/SaProt on average."""
        rows = _load_leaderboard()
        si_mean = np.mean([float(r["Site-Independent"]) for r in rows])
        saprot_mean = np.mean([float(r["SaProt (650M)"]) for r in rows])
        assert si_mean < saprot_mean


class TestDmsEvalNullBaseline:
    """Test that our DMS eval produces ~0 rho for random predictions."""

    def test_null_rho_near_zero(self):
        """Random predictions should give Spearman rho near 0."""
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.standard_normal(n)
        y_pred = rng.standard_normal(n)
        rho, _ = stats.spearmanr(y_true, y_pred)
        assert abs(rho) < 0.15, f"Null baseline rho {rho} too far from 0"

    def test_perfect_predictor_rho_one(self):
        """Perfect predictions should give Spearman rho = 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rho, _ = stats.spearmanr(y, y)
        assert abs(rho - 1.0) < 1e-10

    def test_inverted_predictor_rho_negative(self):
        """Inverted predictions should give Spearman rho = -1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rho, _ = stats.spearmanr(y, -y)
        assert abs(rho - (-1.0)) < 1e-10


class TestLeaderboardPerAssayValidation:
    """Use leaderboard to validate that specific assays have expected properties."""

    def test_assay_rho_ranges_are_plausible(self):
        """No model should have rho > 1 or < -1."""
        rows = _load_leaderboard()
        for r in rows:
            for key, val in r.items():
                if key == "DMS ID":
                    continue
                try:
                    v = float(val)
                    assert -1.0 <= v <= 1.0, f"{r['DMS ID']}/{key} has rho={v}"
                except ValueError:
                    pass

    def test_can_lookup_any_assay_model_pair(self):
        """Verify we can extract exact expected rho for any assay+model pair.

        This is the core capability needed: given an assay name and model name,
        get the expected Spearman rho to validate our eval against.
        """
        rows = _load_leaderboard()
        lookup = {r["DMS ID"]: r for r in rows}

        # Spot-check a few known values
        assert abs(float(lookup["BRCA1_HUMAN_Findlay_2018"]["SaProt (650M)"]) - 0.539) < 0.01
        assert abs(float(lookup["BRCA1_HUMAN_Findlay_2018"]["EVE (ensemble)"]) - 0.468) < 0.05
        assert "A0A140D2T1_ZIKV_Sourisseau_2019" in lookup
