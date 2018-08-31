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


def get_mini_batch_x(mini_batch_size):
    ys = np.random.uniform(size=mini_batch_size)
    xs = ys / (1 - ys)
    return xs


def simulate_once(t):
    mini_batch_size = 5000
    sum_cur = 0
    # n = 0

    while sum_cur <= t:
        # xs = get_mini_batch_x(mini_batch_size=mini_batch_size)
        ys = np.random.uniform(size=mini_batch_size)
        xs = ys / (1 - ys)
        sum_next = sum_cur + np.sum(xs)
        if sum_next > t:
            break
        sum_cur = sum_next

    x_i = 0
    while sum_cur <= t:
        x = xs[x_i]
        x_i += 1
        sum_cur += x  # s3
    numerator = t - (sum_cur - x)
    denominator = x
    ratio = numerator / denominator
    # print(n, sum_cur)
    return ratio


# print(simulate_once(5))


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


def driver_multi_process(t, n):
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


# np.random.seed(1)
# driver_single_process()
#
def main():
    power = 5
    t = 10**power
    n = 10
    # result = driver_multi_process(t=t, n=n)

    np.random.seed(1)
    result = driver_single_process(t=t, n=n)
    result = np.sort(result)
    np.savetxt('result_t{t}_n{n}.csv'.format(t=power, n=n), result, delimiter=',')

    result_len = len(result)
    t = np.arange(0., result_len)
    x1 = [0, result_len]
    y1 = [0, 1]

    # red dashes, blue squares and green triangles
    plt.plot(t, result, 'r.', markersize=1, label='A(t)/C(t)')
    plt.plot(x1, y1, label='y=x')
    plt.xlabel('Sample Size')
    plt.ylabel('Function Value')
    plt.show()


if __name__ == '__main__':
    main()
