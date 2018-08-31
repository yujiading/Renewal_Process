import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial


def get_pdf(y):
    x = y / (1 - y)
    return x


def get_uniform():
    return np.random.uniform()


def simulate_once(t):
    sum_prev = 0
    sum_cur = sum_prev
    n = 0
    while sum_cur < t:
        sum_prev = sum_cur
        y = np.random.uniform()
        x = y / (1 - y)
        sum_cur += x
        n += 1
        # print(n, y, x)
    numerator = t - sum_prev
    denominator = x
    ratio = numerator / denominator
    # print(n, sum_cur)
    return ratio


# print(simulate_once(5))


def mini_batch(i, n,t):
    simulated_data = [simulate_once(t=t) for _ in range(n)]
    return simulated_data


def driver_single_process():
    t = 100000
    n = 1000
    start_time = time.time()
    simulated_data = [simulate_once(t=t) for _ in range(n)]
    simulated_arr = np.asarray(simulated_data)
    kstest = stats.kstest(simulated_arr, 'uniform')
    print(len(simulated_data), kstest)
    print("--- %s seconds ---" % (time.time() - start_time))
    return simulated_arr


def driver_multi_process():
    t = 100000000
    n = 10000
    p = Pool(6)
    start_time = time.time()
    mini_batch_count = 6
    mini_batch_map_items = list(range(mini_batch_count))
    mini_batch_n = round(n / 6)
    nested_result = p.map(partial(mini_batch, n=mini_batch_n, t=t), mini_batch_map_items)
    simulated_data = []
    for result in nested_result:
        simulated_data.extend(result)

    simulated_arr = np.asarray(simulated_data)
    kstest = stats.kstest(simulated_arr, 'uniform')
    print(len(simulated_data), kstest)
    print("--- %s seconds ---" % (time.time() - start_time))
    return simulated_arr

np.random.seed(1)

driver_single_process()

# if __name__ == '__main__':
#     result = driver_multi_process()
#     result = np.sort(result)
#     result_len = len(result)
#     t = np.arange(0., result_len)
#     x1 = [0, result_len]
#     y1 = [0, 1]
#
#     # red dashes, blue squares and green triangles
#     plt.plot(t, result, 'r.', markersize=3)
#     plt.plot(x1, y1)
#     plt.show()
