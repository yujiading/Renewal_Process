import logging
import os
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from simulate_minibatch_binary_search import driver_multi_process


class Driver:
    def __init__(self):
        # ---- Modify ---- #
        # self.is_train = False
        self.is_train = True
        self.power = 7
        self.n_iterations = 1000
        self.iteration_size = 50
        self.n_cpu = 4
        # ---------------- #

        log_file_path = "logs/log_power{power}.txt".format(power=self.power)
        self.logger = self.__get_logger(log_file_path)
        self.result_file_path = "results/result_power{power}.csv".format(power=self.power)

    def __get_logger(self, log_file_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if self.is_train:
            # create file handler which logs even debug messages
            fh = logging.FileHandler(log_file_path, mode='a+')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
        return logger

    def load_old_results(self, file_path):
        try:
            old_results = np.loadtxt(file_path)
        except:
            self.logger.warning('File not found, ignoring old results', exc_info=True)
            old_results = []
        return old_results

    def plot_results(self, result_arr):
        result_arr = np.sort(result_arr)
        result_len = len(result_arr)
        t = np.arange(0., result_len)
        x1 = [0, result_len]
        y1 = [0, 1]

        # red dashes, blue squares and green triangles
        plt.plot(t, result_arr, 'r.', markersize=1, label='A(t)/C(t)')
        plt.plot(x1, y1, label='y=x')
        plt.xlabel('Sample Size')
        plt.ylabel('Function Value')
        plt.show()

    def train_one_iteration(self, process_pool):
        t = 10 ** self.power
        train_time = 0

        if self.is_train:
            start_time = time.time()
            simulated_data_list = driver_multi_process(t=t, n=self.iteration_size,
                                                       process_pool=process_pool)
            train_time = time.time() - start_time
            old_results = self.load_old_results(self.result_file_path)
            simulated_arr = np.asarray(simulated_data_list)
            simulated_arr = np.concatenate((old_results, simulated_arr))
            np.savetxt(self.result_file_path, simulated_arr, delimiter=',')
        else:
            simulated_arr = self.load_old_results(self.result_file_path)

        ks_result = stats.kstest(simulated_arr, 'uniform')
        self.logger.info("train_time {train_time:4.2f}s,num_sample {num_sample}, {ks_result}".format(
            train_time=train_time, num_sample=len(simulated_arr), ks_result=ks_result
        ))

        return simulated_arr

    def seeder(self):
        seed_id = os.getpid()
        np.random.seed(seed_id)
        print("seed", seed_id)

    def run(self):
        process_pool = Pool(self.n_cpu, initializer=self.seeder)

        for _ in range(self.n_iterations):
            iteration_results = self.train_one_iteration(process_pool=process_pool)
            if not self.is_train:
                break
        self.plot_results(result_arr=iteration_results)


if __name__ == '__main__':
    np.random.seed(0)
    driver = Driver()
    driver.run()
