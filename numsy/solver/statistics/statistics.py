from __future__ import annotations

import math
import statistics

from enum import Enum
from typing import Any, Callable, Concatenate, ParamSpec, Generic, TypeVar

from numsy.solver.errors import EmptyDataError, SlicingError, InsufficientDataError
from . import _quantiles


P = ParamSpec("P")
T = TypeVar("T", bound=float)


class QuantileEnum(Enum):
    R_1 = "r1"
    R_2 = "r2"
    R_3 = "r3"
    R_4 = "r4"
    R_5 = "r5"
    R_6 = "r6"
    R_7 = "r7"
    R_8 = "r8"
    R_9 = "r9"


class DataTypeEnum(Enum):
    sample = "sample"
    population = "population"


def not_empty(func: Callable[Concatenate[Data, P], Any]):
    def inner(self: Data, *args: P.args, **kwargs: P.kwargs):
        if isinstance(self, Data) and len(self) == 0:
            raise EmptyDataError
        return func(self, *args, **kwargs)
    return inner


class Data(Generic[T]):
    def __init__(self, data: list[T]) -> None:
        self.data = data

    @property
    @not_empty
    def mean(self) -> float:
        """Calculates the mean (average) of the dataset.

        Returns:
            float: The mean of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return statistics.mean(data=self.data)

    @property
    @not_empty
    def median(self) -> float:
        """Calculates the median of the dataset.

        Returns:
            float: The median of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return statistics.median(data=self.data)

    @property
    @not_empty
    def median_low(self) -> float:
        """Calculates the low median of the dataset.

        The low median is the smaller of the two middle values in a dataset with an even number of elements.

        Returns:
            float: The low median of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return statistics.median_low(data=self.data)

    @property
    @not_empty
    def median_high(self) -> float:
        """Calculates the high median of the dataset.

        The high median is the larger of the two middle values in a dataset with an even number of elements.

        Returns:
            float: The high median of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return statistics.median_high(data=self.data)

    @property
    @not_empty
    def mode(self) -> float:
        """Calculates the mode of the dataset.

        The mode is the most frequently occurring value in the dataset.
        If multiple values have the same frequency, the first occurring value is returned.

        Returns:
            float: The mode of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return statistics.mode(data=self.data)

    @property
    def modes(self) -> list[float]:
        """Calculates the modes (all most frequently occurring values) of the dataset.

        Returns:
            list[float]: A list of modes in the dataset. Returns an empty list if the dataset is empty.
        """
        return statistics.multimode(data=self.data)

    @property
    @not_empty
    def range(self) -> float:
        """Calculates the range of the dataset.

        The range is the difference between the maximum and minimum values in the dataset.

        Returns:
            float: The range of the dataset.

        Raises:
            `EmptyDataError`: If the dataset is empty.
        """
        return max(self.data) - min(self.data)

    @not_empty
    def quantiles(self, slices: int = 4, method: QuantileEnum = QuantileEnum.R_6) -> list[float]:
        """Divides the dataset into equal slices and returns the quantiles.

        Quantiles are the cut points that divide the dataset into equally sized intervals based on the chosen method.
        See https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample for more information.
        Examples:
            - **10 slices** correspond to deciles.
            - **4 slices (default)** correspond to quartiles.
            - **2 slices** correspond to the median.

        Parameters:
            slices (int): The number of equal slices to divide the dataset into. Must be greater than 1.
            method (`QuantileEnum`): The method to compute quantiles. Defaults to `QuantileEnum.R_6`.

        Returns:
            list[float]: A list of quantile values.

        Raises:
            ValueError: If the number of slices is less than or equal to 1.
            SlicingError: If the number of slices exceeds the dataset size + 2.
            EmptyDataError: If the dataset is empty.
        """
        if slices <= 1:
            raise ValueError("Cannot slice 0 times.")
        elif slices >= len(self) + 2:
            raise SlicingError(slices=slices, size=len(self))
        res: list[float] = []
        # If we have `i/slices` as `4/4`, several functions will raise IndexError
        for i in range(1, slices):
            res.append(getattr(_quantiles, f"quantile_{method.value}")(sorted(self.data), i/slices))
        return res

    def standard_deviation(self, mean: float | None = None, _type: DataTypeEnum = DataTypeEnum.population):
        """Calculates the standard deviation of the dataset.

        The standard deviation measures the dispersion of data points around the mean.
        This method supports both population and sample standard deviation.

        Parameters:
            mean (float | None): The mean of the dataset. If not provided, it is computed from the data.
            _type (`DataTypeEnum`): The type of standard deviation to calculate:
                - `DataTypeEnum.sample`: Calculates the sample standard deviation.
                - `DataTypeEnum.population` (default): Calculates the population standard deviation.

        Returns:
            float: The calculated standard deviation.

        Raises:
            `InsufficientDataError`: If the dataset contains fewer than two data points.
        """
        if len(self) < 2:
            raise InsufficientDataError(data_points=2)
        if _type is DataTypeEnum.sample:
            return statistics.stdev(data=self.data, xbar=mean)
        return statistics.pstdev(data=self.data, mu=mean)

    def variance(self, mean: float | None = None, _type: DataTypeEnum = DataTypeEnum.population):
        """Calculates the variance of the dataset.

        Variance represents the average squared deviation of data points from the mean.
        This method supports both population and sample variance.

        Parameters:
            mean (float | None): The mean of the dataset. If not provided, it is computed from the data.
            _type (`DataTypeEnum`): The type of variance to calculate:
                - `DataTypeEnum.sample`: Calculates the sample variance.
                - `DataTypeEnum.population` (default): Calculates the population variance.

        Returns:
            float: The calculated variance.

        Raises:
            `InsufficientDataError`: If the dataset contains fewer than two data points.
        """

        if len(self) < 2:
            raise InsufficientDataError(data_points=2)
        if _type is DataTypeEnum.sample:
            return statistics.variance(data=self.data, xbar=mean)
        return statistics.pvariance(data=self.data, mu=mean)

    @not_empty
    def mean_absolute_deviation(self, mean: float | None = None):
        """Calculates the Mean Absolute Deviation (MAD) of the dataset.

        The Mean Absolute Deviation measures the average absolute difference between each data point
        and the mean of the dataset. This provides an idea of how spread out the data values are.

        Parameters:
            mean (float | None): The mean of the dataset. If not provided, the mean is calculated
                from the data.

        Returns:
            float: The mean absolute deviation of the dataset.

        Raises:
            EmptyDataError: If the dataset is empty.
        """
        m = mean or self.mean
        return sum(abs(i - m) for i in self.data)/len(self)

    def real_quartiles(self) -> tuple[float, float, float]:
        """Calculates the first, second, and third quartiles (Q1, Q2, Q3) of the dataset.

        Quartiles divide the data into four equal parts:
        - Q1 (lower quartile): The median of the lower half of the dataset.
        - Q2 (median): The median of the entire dataset.
        - Q3 (upper quartile): The median of the upper half of the dataset.

        Returns:
            tuple[float, float, float]: A tuple containing Q1, Q2, and Q3.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        if len(self) < 3:
            raise InsufficientDataError(data_points=4)
        if len(self) % 2 == 0:
            Q1 = Data(self.data[:math.ceil(len(self) / 2)]).median
        else:
            Q1 = Data(self.data[:math.floor(len(self) / 2)]).median
        Q2 = self.median
        Q3 = Data(self.data[math.ceil(len(self) / 2):]).median
        return Q1, Q2, Q3

    def quartile_mean(self) -> float:
        """Calculates the quartile mean.

        The quartile mean is the average of the first and third quartiles (Q1 and Q3).

        Returns:
            float: The quartile mean.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        Q1, _, Q3 = self.real_quartiles()
        return (Q1 + Q3) / 2

    def quartile_deviation(self) -> float:
        """Calculates the quartile deviation (semi-interquartile range).

        The quartile deviation measures the spread of the middle 50% of the data.

        Returns:
            float: The quartile deviation.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        Q1, _, Q3 = self.real_quartiles()
        return (Q3 - Q1) / 2

    def expanse(self) -> float:
        """Calculates the interquartile range (IQR).

        The IQR is the difference between the third quartile (Q3) and the first quartile (Q1).
        It represents the range of the middle 50% of the data.

        Returns:
            float: The interquartile range.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        Q1, _, Q3 = self.real_quartiles()
        return Q3 - Q1

    def average_of_three(self) -> float:
        """Calculates the average of three quartiles (Q1, Q2, Q3).

        The formula used is: (Q1 + 2 * Q2 + Q3) / 4.

        Returns:
            float: The weighted average of the three quartiles.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        Q1, Q2, Q3 = self.real_quartiles()
        return (Q1 + 2 * Q2 + Q3) / 4

    def tukey_fences(self) -> list[float]:
        """Identifies outliers using Tukey's fences method.

        Tukey's fences classify values as outliers if they are:
        - Less than Q1 - 1.5 * IQR
        - Greater than Q3 + 1.5 * IQR

        Returns:
            list[float]: A list of outliers.

        Raises:
            InsufficientDataError: If the dataset has fewer than 3 elements.
        """
        Q1, _, Q3 = self.real_quartiles()
        lb = Q1 - 1.5 * self.expanse()
        ub = Q3 + 1.5 * self.expanse()
        return [x for x in self.data if x < lb or x > ub]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> T:
        return self.data[item]
