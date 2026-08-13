"""Audit regression tests: standardized Student-t VaR scale (P0 fix).

arch's Student-t innovation is a unit-variance standardized t distribution.
The old implementation used the unstandardized scipy.stats.t.ppf directly,
overstating VaR magnitude by sqrt(nu/(nu-2)) (about 22% at nu=6).
"""

from __future__ import annotations

import numpy as np
import pytest
from arch.univariate import StudentsT
from scipy import stats

from spyvar.models.garch import GARCHFamily
from spyvar.rolling import make_window

NUS = [4.5, 6.0, 10.0, 30.0]
ALPHAS = [0.01, 0.05, 0.10]


def test_arch_studentst_matches_scaled_scipy():
    """arch StudentsT().ppf equals scipy t quantile times sqrt((nu-2)/nu)."""
    for nu in NUS:
        q_arch = np.array(StudentsT().ppf(ALPHAS, nu))
        q_scaled = stats.t.ppf(ALPHAS, nu) * np.sqrt((nu - 2.0) / nu)
        np.testing.assert_allclose(q_arch, q_scaled, rtol=1e-12, atol=1e-14)


def test_standardized_t_has_unit_variance():
    """The standardized t used by arch has variance 1 (by construction)."""
    rng = np.random.default_rng(0)
    for nu in (6.0, 10.0):
        z = stats.t.rvs(nu, size=1_000_000, random_state=rng)
        z_std = z / np.sqrt(nu / (nu - 2.0))
        assert z_std.var() == pytest.approx(1.0, rel=0.01)
        assert z_std.mean() == pytest.approx(0.0, abs=0.01)


def test_old_implementation_was_overstated():
    """Regression guard: the unstandardized quantile is systematically too large.

    At nu=6 the unscaled 1% quantile is about 3.143 while the correct
    standardized one is about 2.566 (a ~22% relative error).
    """
    for nu in (4.5, 6.0, 10.0):
        naive = stats.t.ppf(ALPHAS, nu)
        correct = np.array(StudentsT().ppf(ALPHAS, nu))
        assert np.all(naive < correct)  # naive is more negative -> overstates risk
        rel = np.abs((naive - correct) / correct)
        assert np.all(rel > 0.02)  # materially different, never silently close


def test_garch_family_uses_standardized_quantiles(synth_df_small):
    """End-to-end: GARCH-t / GJR-t VaR must be consistent with StudentsT().ppf."""
    w = make_window(synth_df_small, 500, 300, None, seed=42, model_config={})
    for model in (GARCHFamily(model_id="M1"), GARCHFamily(model_id="M4", o=1)):
        fc = model.fit(w)
        assert fc.fit_status == "ok"
        nu = fc.meta["nu"]
        sigma = fc.meta["sigma_next"]
        mu = fc.q_005 - sigma * float(StudentsT().ppf([0.05], nu)[0])
        # mu back-solved from one quantile must be consistent across tails
        for a, key in ((0.01, "q_001"), (0.10, "q_010")):
            q_expect = mu + sigma * float(StudentsT().ppf([a], nu)[0])
            assert getattr(fc, key) == pytest.approx(q_expect, rel=1e-10)


def test_gaussian_garch_unaffected(synth_df_small):
    """Gaussian GARCH uses norm.ppf; it was never affected by the t-scale bug."""
    w = make_window(synth_df_small, 500, 300, None, seed=42, model_config={})
    fc = GARCHFamily(model_id="M1_gauss", dist="normal").fit(w)
    assert fc.fit_status == "ok"
    sigma = fc.meta["sigma_next"]
    mu = fc.q_005 - sigma * stats.norm.ppf(0.05)
    assert fc.q_001 == pytest.approx(mu + sigma * stats.norm.ppf(0.01), rel=1e-10)
