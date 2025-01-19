from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .statistics import Data


def quantile_r1(data: Data, p: float):
    """Type R-1: Inverse of the empirical distribution function."""
    return data[int(p * len(data)) - 1]


def quantile_r2(data: Data, p: float):
    """Type R-2: Empirical CDF with averaging at discontinuities."""
    slice_ = p * len(data)
    upper = int(slice_)
    lower = upper - 1
    if slice_ % 1 == 0:
        return data[lower]
    return (data[lower] + data[upper]) / 2


def quantile_r3(data: Data, p: float):
    """Type R-3: Nearest observation."""
    return data[round(p * len(data)) - 1]


def quantile_r4(data: Data, p: float):
    """Type R-4: Linear interpolation of the inverse empirical CDF."""
    slice_ = p * len(data)
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])


def quantile_r5(data: Data, p: float):
    """Type R-5: Piecewise linear function."""
    slice_ = p * len(data) + 0.5
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])


def quantile_r6(data: Data, p: float):
    """Type R-6: Linear interpolation of expectations for uniform distribution."""
    slice_ = (len(data) + 1) * p
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])


def quantile_r7(data: Data, p: float):
    """Type R-7: Default method in many statistical packages."""
    slice_ = (len(data) - 1) * p + 1
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])


def quantile_r8(data: Data, p: float):
    """Type R-8: Linear interpolation of approximate medians."""
    slice_ = (len(data) + 1 / 3) * p + (1 / 3)
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])


def quantile_r9(data: Data, p: float):
    """Type R-9: Quantiles unbiased for normal distributions."""
    slice_ = (len(data) + 1 / 4) * p + (3 / 8)
    upper = int(slice_)
    lower = upper - 1
    return data[lower] + (slice_ - int(slice_)) * (data[upper] - data[lower])
