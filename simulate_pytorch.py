import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import torch
import torch as th


uniform = torch.distributions.uniform.Uniform(0, 1)
sample_shape = torch.Size(torch.tensor([1, 5000]))

def get_mini_batch_x(sample_shape) -> torch.Tensor:
    ys = uniform.sample(sample_shape=sample_shape)
    xs = ys / (torch.ones(sample_shape) - ys)
    return xs


def simulate_once(t: int):
    t_tensor = torch.tensor([1.0 * t])
    sum_cur = torch.tensor([0.])

    # n = 0

    while sum_cur < t_tensor:
        xs = get_mini_batch_x(sample_shape=sample_shape)
        mini_batch_sum = xs.sum(1)
        sum_next = sum_cur + mini_batch_sum
        if sum_next >= t_tensor:
            break
        sum_cur = sum_next

    x_i = torch.tensor([0])
    one = torch.tensor([1])
    while sum_cur < t_tensor:
        x = xs[0][x_i]
        x_i += one
        sum_cur += x  # s3
    numerator = t_tensor - (sum_cur - x)
    denominator = x
    ratio = numerator / denominator
    ratio_float = ratio.numpy()[0]
    # print(n, sum_cur)
    return ratio_float


# print(simulate_once(5))


# def mini_batch(i, n, t):
#     simulated_data = [simulate_once(t=t) for _ in range(n)]
#     return simulated_data


def driver_single_process(t, n):
    start_time = time.time()
    simulated_data = [simulate_once(t=t) for _ in range(n)]
    simulated_arr = np.asarray(simulated_data)
    kstest = stats.kstest(simulated_arr, 'uniform')
    print(len(simulated_data), kstest)
    print("--- %s seconds ---" % (time.time() - start_time))
    return simulated_arr


# def driver_multi_process():
#     # t = 100000000
#     # n = 10000
#     t = 100000
#     n = 1000
#     p = Pool(6)
#     start_time = time.time()
#     mini_batch_count = 6
#     mini_batch_map_items = list(range(mini_batch_count))
#     mini_batch_n = round(n / 6)
#     nested_result = p.map(partial(mini_batch, n=mini_batch_n, t=t), mini_batch_map_items)
#     simulated_data = []
#     for result in nested_result:
#         simulated_data.extend(result)
#
#     simulated_arr = np.asarray(simulated_data)
#     kstest = stats.kstest(simulated_arr, 'uniform')
#     print(len(simulated_data), kstest)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     return simulated_arr


# np.random.seed(1)
# driver_single_process()

if __name__ == '__main__':
    t = 100000
    n = 100
    # result = driver_multi_process()
    result = driver_single_process(t=t, n=n)
    result = np.sort(result)
    np.savetxt('result_t{t}_n{n}.csv'.format(t=t, n=n), result, delimiter=',')

    result_len = len(result)
    t = np.arange(0., result_len)
    x1 = [0, result_len]
    y1 = [0, 1]

    # red dashes, blue squares and green triangles
    plt.plot(t, result, 'r.', markersize=1, label='A(t)/C(t)')
    plt.plot(x1, y1, label='y=x')
    plt.xlabel('Sample')
    plt.ylabel('Function Value')
    plt.show()
