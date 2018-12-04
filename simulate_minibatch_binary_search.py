import math
import numpy as np
from scipy import stats
import time
from functools import partial


def get_pdf(y):
    x = y / (1 - y)
    return x


def get_uniform():
    return np.random.uniform()


def rolling_random_array(random_array, remove_count, alpha):
    """

    :param random_array: array of random number
    :param remove_count: first remove_count elements should be removed, and new random numbers should be appended
    :param alpha:
    :return:
    """
    remaining_items = random_array[remove_count:]
    added_elements = get_mini_batch_x(mini_batch_size=remove_count, alpha=alpha)
    return np.concatenate([remaining_items, added_elements])


def binary_search(xs, sum_cur, t):
    """

    :param xs:
    :param sum_cur:
    :param t:
    :return: (last item before sum reach t, sum until and including the last item before reach t)
    """
    left = 0
    right = len(xs) - 1
    while left <= right:
        mid = math.floor((left + right) / 2)

        sum_left_to_mid = np.sum(xs[left:mid + 1])
        next_sum = sum_left_to_mid + sum_cur
        if next_sum <= t:
            # item to find not on the left of mid
            left = mid + 1
            sum_cur = next_sum
        else:
            right = mid - 1
    mid = math.floor((left + right) / 2)

    return mid, sum_cur


def get_mini_batch_x(mini_batch_size, alpha):
    ys = np.random.uniform(size=mini_batch_size)
    xs = (1 / ys - 1) ** (1 / alpha)
    return xs


xs_from_previous_run = None


def simulate_once(t):
    global xs_from_previous_run
    mini_batch_size = 5000
    sum_cur = 0
    n = 0
    alpha = 0.9
    xs = None

    while sum_cur <= t:
        if xs_from_previous_run is not None:
            xs = xs_from_previous_run
            xs_from_previous_run = None
        else:
            xs = get_mini_batch_x(mini_batch_size=mini_batch_size, alpha=alpha)
        sum_next = sum_cur + np.sum(xs)
        if sum_next > t:
            break
        sum_cur = sum_next
        n += len(xs)

    idx, sum_incl_idx = binary_search(xs, sum_cur, t)
    x = xs[idx + 1]

    # numerator = t - sum_incl_idx
    numerator = t - (sum_incl_idx + x) + x
    denominator = x
    ratio = (numerator / denominator) ** alpha

    n += idx + 1
    # print("{:,}".format(n), sum_cur)
    remove_count = idx + 2
    xs_from_previous_run = rolling_random_array(
        random_array=xs,
        remove_count=remove_count,
        alpha=alpha
    )

    return ratio


def mini_batch(i, n, t):
    simulated_data = [simulate_once(t=t) for _ in range(n)]
    return simulated_data


def driver_single_process(t, n):
    start_time = time.time()
    simulated_data = [simulate_once(t=t) for _ in range(n)]
    simulated_arr = np.asarray(simulated_data)
    kstest = stats.kstest(simulated_arr, 'uniform')
    print(len(simulated_data), kstest)
    print("--- %s seconds ---" % (time.time() - start_time))
    return simulated_arr


def driver_multi_process(process_pool, t, n):
    mini_batch_count = 6
    mini_batch_map_items = list(range(mini_batch_count))
    mini_batch_n = round(n / 6)
    nested_result = process_pool.map(partial(mini_batch, n=mini_batch_n, t=t),
                                     mini_batch_map_items)
    simulated_data = []
    for result in nested_result:
        simulated_data.extend(result)

    return simulated_data

# # np.random.seed(1)
# # result = np.random.uniform(size=50000)
# # # result = np.sort(result)
# # result = stats.uniform.rvs(size=50000)
# # # print(result)
# # kstest = stats.kstest(result, 'uniform')
# # print(kstest)
#
# # np.random.seed(1)
# # driver_single_process()
# #
# def main():
#     # power = 5
#     # t = 10**power
#     # n = 10000
#     # # result = driver_multi_process(t=t, n=n)
#     #
#     # # np.random.seed(1)
#     # result = driver_single_process(t=t, n=n)
#     # result = np.sort(result)
#     # np.savetxt('result_t{t}_n{n}.csv'.format(t=power, n=n), result, delimiter=',')
#
#     result = np.random.uniform(size=50000)
#     result = np.sort(result)
#
#     kstest = stats.kstest(result, 'uniform')
#     print(kstest)
#
#     result_len = len(result)
#     t = np.arange(0., result_len)
#     x1 = [0, result_len]
#     y1 = [0, 1]
#
#     # red dashes, blue squares and green triangles
#     plt.plot(t, result, 'r.', markersize=1, label='A(t)/C(t)')
#     plt.plot(x1, y1, label='y=x')
#     plt.xlabel('Sample Size')
#     plt.ylabel('Function Value')
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
