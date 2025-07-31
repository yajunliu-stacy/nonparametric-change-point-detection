import numpy as np
from numpy import random
from tqdm import tqdm

class ThresholdGenerator:
    def __init__(self, grid, length):
        self.grid = grid
        self.length = length

    def threshold_logit(self, X_t, c=0):
        X_t_trans = np.clip(X_t, c, 1 - c)
        return np.log(X_t_trans / (1 - X_t_trans))

    def compute_quantile_map(self, x):
        quantiles = [i / (self.grid + 1) for i in range(1, self.grid + 1)]
        x_quants = np.quantile(x, quantiles, method='nearest')
        return quantiles, x_quants

    def compute_variance(self, x, x_quantile):
        bool_series = np.array([int(z <= x_quantile) for z in x])
        cov = np.var(bool_series)
        for lag in range(1, self.length + 1):
            cov_lag = 2 * np.cov(bool_series[:-lag], bool_series[lag:])[0][1]
            cov += cov_lag
        return max(0, cov)

    def compute_gaussian_map(self, x, quantiles, x_quants):
        cov_list = []
        prev_quantile = None
        prev_cov = None
        for x_quant in x_quants:
            if x_quant == prev_quantile:
                cov_list.append(prev_cov)
            else:
                prev_quantile = x_quant
                prev_cov = self.compute_variance(x, x_quant)
                cov_list.append(prev_cov)
        return list(zip(quantiles, x_quants, cov_list))

    def get_quantile_mapping(self, x):
        quantiles, x_quants = self.compute_quantile_map(x)
        quantile_map = self.compute_gaussian_map(x, quantiles, x_quants)
        return quantiles, x_quants, quantile_map

    def compute_covariance(self, x, x_quant1, x_quant2):
        b1 = np.array([int(z <= x_quant1) for z in x])
        b2 = np.array([int(z <= x_quant2) for z in x])
        cov = np.cov(b1, b2)[0][1]
        for lag in range(1, self.length + 1):
            cov += np.cov(b1[:-lag], b2[lag:])[0][1]
            cov += np.cov(b2[:-lag], b1[lag:])[0][1]
        return cov

    def get_covariance_matrix(self, x, x_quants):
        n = len(x_quants)
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = self.compute_covariance(x, x_quants[i], x_quants[j])
        return cov_matrix

    def generate_wiener_process(self, m, n, sigma):
        l = len(sigma)
        mu_0 = np.zeros(l)
        w2 = random.multivariate_normal(mu_0, sigma)
        paths = [[] for _ in range(l)]
        for _ in range(int(n * m)):
            x = random.multivariate_normal(mu_0, sigma)
            for k in range(l):
                paths[k].append(x[k] - w2[k] / np.sqrt(m))
        cumsum_paths = np.cumsum(paths, axis=1) / np.sqrt(m)
        return cumsum_paths

    def generate_wiener_with_gamma(self, gamma_values, sigma, A, m0, n):
        wiener = self.generate_wiener_process(m0, n, sigma).T
        steps = wiener.shape[0]
        result = np.zeros((len(gamma_values), steps))
        for idx, gamma in enumerate(gamma_values):
            for i in range(steps):
                scale = (1 + (i + 1) / m0) ** -2 * ((i + 1) / (m0 + 1 + i)) ** (-2 * gamma)
                result[idx, i] = scale * np.dot(np.dot(wiener[i], A), wiener[i])
        return result

    def threshold_generation_given_seq(self, y_list, n, num_iterations, A=None, m0=50, gamma_values=[0, 0.25, 0.4], threshold_values = [0.9, 0.95, 0.975, 0.99]):
        quantiles, x_quants, quantile_map = self.get_quantile_mapping(y_list)
        cov_matrix = self.get_covariance_matrix(y_list, x_quants)

        if A is None:
            A = np.linalg.inv(cov_matrix)

        sup_list = []
        for _ in tqdm(range(num_iterations)):
            cumsum_matrix = self.generate_wiener_with_gamma(gamma_values, cov_matrix, A, m0, n)
            sup_values = np.max(cumsum_matrix, axis=1)
            sup_list.append(sup_values)

        thresholds = np.quantile(sup_list, threshold_values, axis=0, method='nearest')
        return thresholds, quantiles, x_quants, quantile_map, A