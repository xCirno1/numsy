import pytest

from numsy.solver.statistics import Data, DataTypeEnum
from numsy.solver.errors import EmptyDataError, SlicingError

DATASET_1 = [7, 3, 5]
DATASET_2 = [2, 8, 1, 3, 6]
DATASET_3 = [9, 1, 4, 4, 2, 6, 8, 3]
DATASET_4 = [5, 7, 1, 4]

DATASET_5 = [5, 1, 2, 4, 4, 8]
DATASET_6 = [1, 9, 6, 3, 4, 6, 2]
DATASET_7 = [5, 3, 2, 4, 6, 3, 1, 5, 5]

def test_data_operations_on_empty_data():
    empty_data = Data([])
    operations = ['mean', 'median', 'mode']

    for operation in operations:
        with pytest.raises(EmptyDataError):
            getattr(empty_data, operation)

def test_data_mean():
    assert Data(DATASET_1).mean == pytest.approx(5)
    assert Data(DATASET_2).mean == pytest.approx(4)
    assert Data(DATASET_3).mean == pytest.approx(4.625)
    assert Data(DATASET_4).mean == pytest.approx(4.25)

def test_data_median():
    assert Data(DATASET_1).median == 5
    assert Data(DATASET_2).median == 3
    assert Data(DATASET_3).median == 4
    assert Data(DATASET_4).median == 4.5

def test_data_median_low_value():
    assert Data(DATASET_1).median_low == 5
    assert Data(DATASET_2).median_low == 3
    assert Data(DATASET_3).median_low == 4
    assert Data(DATASET_4).median_low == 4

def test_data_median_high_value():
    assert Data(DATASET_1).median_high == 5
    assert Data(DATASET_2).median_high == 3
    assert Data(DATASET_3).median_high == 4
    assert Data(DATASET_4).median_high == 5

def test_data_mode():
    assert Data(DATASET_1).mode == 7
    assert Data(DATASET_2).mode == 2
    assert Data(DATASET_3).mode == 4
    assert Data(DATASET_4).mode == 5

def test_data_modes():
    assert Data(DATASET_1).modes == [7, 3, 5]
    assert Data(DATASET_2).modes == [2, 8, 1, 3, 6]
    assert Data(DATASET_3).modes == [4]


def test_range():
    assert Data(DATASET_5).range == 7
    assert Data(DATASET_6).range == 8
    assert Data(DATASET_7).range == 5

def test_quantiles():
    assert Data(DATASET_5).quantiles(4) == [1.75, 4.0, 5.75]
    assert Data(DATASET_6).quantiles(4) == [2.0, 4.0, 6.0]
    assert Data(DATASET_7).quantiles(4) == [2.5, 4.0, 5.0]
    assert Data(DATASET_5).quantiles(2) == [4.0]
    assert Data(DATASET_6).quantiles(2) == [4.0]

    with pytest.raises(ValueError):
        _ = Data(DATASET_5).quantiles(1)
    with pytest.raises(SlicingError):
        _ = Data(DATASET_5).quantiles(20)

def test_standard_deviation():
    assert Data(DATASET_5).standard_deviation() == pytest.approx(2.23606797749979)
    assert Data(DATASET_6).standard_deviation() == pytest.approx(2.5555062599997598)
    assert Data(DATASET_7).standard_deviation(_type=DataTypeEnum.sample) == pytest.approx(1.6414763002993509)
    assert Data(DATASET_5).standard_deviation(_type=DataTypeEnum.sample) == pytest.approx(2.449489742783178)


def test_variance():
    assert Data(DATASET_5).variance() == 5
    assert Data(DATASET_6).variance() == pytest.approx(6.530612244897959)
    assert Data(DATASET_7).variance(_type=DataTypeEnum.sample) == pytest.approx(2.6944444444444446)
    assert Data(DATASET_5).variance(_type=DataTypeEnum.sample) == 6

def test_real_quartiles():
    assert Data([1, 2, 3, 4, 5, 6]).real_quartiles() == (2, 3.5, 5)
    assert Data([1, 2, 3, 4, 5, 6, 7]).real_quartiles() == (2, 4, 6)
    assert Data([1, 2, 3, 4, 5, 6, 7, 8]).real_quartiles() == (2.5, 4.5, 6.5)
    assert Data([1, 2, 3, 4, 5, 6, 7, 8, 9]).real_quartiles() == (2.5, 5, 7.5)
