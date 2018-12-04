import numpy as np

from simulate_minibatch_binary_search import get_mini_batch_x
from simulate_minibatch_binary_search import rolling_random_array


def test_rolling_random_array():
    np.random.seed(0)
    array_size = 10
    remove_count = 3
    alpha = 0.9
    initial_array = get_mini_batch_x(array_size, alpha)
    result_array = rolling_random_array(initial_array, remove_count, alpha)

    np.random.seed(0)
    expected_result = get_mini_batch_x(array_size + remove_count, alpha)[remove_count:]
    assert len(expected_result) == array_size
    assert list(expected_result) == list(result_array)
