import numpy as np
from scipy import stats


def tir(glucose_trace):
    """Time-In-Range: % of time in [70, 180] mg/dL."""
    arr = np.asarray(glucose_trace)
    return 100.0 * np.mean((arr >= 70) & (arr <= 180))


def tbr(glucose_trace):
    """Time Below Range: % of time < 70 mg/dL (hypoglycemia)."""
    arr = np.asarray(glucose_trace)
    return 100.0 * np.mean(arr < 70)


def tar(glucose_trace):
    """Time Above Range: % of time > 180 mg/dL (hyperglycemia)."""
    arr = np.asarray(glucose_trace)
    return 100.0 * np.mean(arr > 180)


def lbgi(glucose_trace):
    """Low Blood Glucose Index (Magni et al. 2011)."""
    arr = np.asarray(glucose_trace, dtype=float)
    f = 1.509 * (np.log(arr + 1e-6) ** 1.084 - 5.381)
    rl = 10 * f ** 2 * (f < 0)
    return float(np.mean(rl))


def hbgi(glucose_trace):
    """High Blood Glucose Index (Magni et al. 2011)."""
    arr = np.asarray(glucose_trace, dtype=float)
    f = 1.509 * (np.log(arr + 1e-6) ** 1.084 - 5.381)
    rh = 10 * f ** 2 * (f > 0)
    return float(np.mean(rh))


def mean_glucose(glucose_trace):
    return float(np.mean(glucose_trace))


def cv_glucose(glucose_trace):
    """Coefficient of variation (%)."""
    arr = np.asarray(glucose_trace)
    m = np.mean(arr)
    return float(100.0 * np.std(arr) / m) if m > 0 else 0.0


def failure_rate(episodes):
    """Fraction of episodes where glucose went < 40 or > 600 mg/dL."""
    failures = sum(
        1 for ep in episodes if any(g < 40 or g > 600 for g in ep)
    )
    return 100.0 * failures / len(episodes) if episodes else 0.0


def compute_episode_metrics(glucose_trace):
    """Return a dict of all metrics for one episode."""
    return {
        'tir':  tir(glucose_trace),
        'tbr':  tbr(glucose_trace),
        'tar':  tar(glucose_trace),
        'lbgi': lbgi(glucose_trace),
        'hbgi': hbgi(glucose_trace),
        'mean': mean_glucose(glucose_trace),
        'cv':   cv_glucose(glucose_trace),
    }


def compare(metrics_a, metrics_b, metric_name):
    """Mann-Whitney U test + Cohen's r effect size for one metric.

    metrics_a / metrics_b: lists of per-episode scalar values.
    Returns (median_a, iqr_a, median_b, iqr_b, p_value, cohens_r).
    """
    a = np.asarray(metrics_a)
    b = np.asarray(metrics_b)

    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    n = len(a) + len(b)
    z = stats.norm.ppf(p / 2)
    r = abs(z) / np.sqrt(n)

    def iqr(x):
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    return {
        'metric': metric_name,
        'median_a': float(np.median(a)),
        'iqr_a':    iqr(a),
        'median_b': float(np.median(b)),
        'iqr_b':    iqr(b),
        'p_value':  float(p),
        'cohens_r': float(r),
    }
