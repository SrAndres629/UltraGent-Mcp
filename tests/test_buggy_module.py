# tests/test_buggy_module.py
import pytest
from buggy_module import sum_positive_numbers
from unittest.mock import patch, MagicMock

@pytest.fixture
def numbers():
    return [1, 2, 3, 4, 5]

@pytest.fixture
def negative_numbers():
    return [-1, -2, -3, -4, -5]

@pytest.fixture
def mixed_numbers():
    return [-1, 2, -3, 4, -5]

def test_sum_positive_numbers_empty_list():
    assert sum_positive_numbers([]) == 0

def test_sum_positive_numbers_all_positive(numbers):
    assert sum_positive_numbers(numbers) == sum(numbers)

def test_sum_positive_numbers_all_negative(negative_numbers):
    assert sum_positive_numbers(negative_numbers) == 0

def test_sum_positive_numbers_mixed(mixed_numbers):
    expected_result = sum(num for num in mixed_numbers if num > 0)
    assert sum_positive_numbers(mixed_numbers) == expected_result

def test_sum_positive_numbers_with_zero():
    numbers = [1, 2, 0, 4, 5]
    expected_result = sum(num for num in numbers if num > 0)
    assert sum_positive_numbers(numbers) == expected_result

def test_sum_positive_numbers_with_non_numeric_values():
    numbers = [1, 'a', 3, None, 5]
    with pytest.raises(TypeError):
        sum_positive_numbers(numbers)

@patch('buggy_module.sum')
def test_sum_positive_numbers_sum_function(mock_sum):
    numbers = [1, 2, 3, 4, 5]
    sum_positive_numbers(numbers)
    mock_sum.assert_called_once_with(numbers)

def test_sum_positive_numbers_invalid_input():
    with pytest.raises(TypeError):
        sum_positive_numbers('not a list')

def test_sum_positive_numbers_non_iterable_input():
    with pytest.raises(TypeError):
        sum_positive_numbers(123)