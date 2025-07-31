import numpy as np

class NonparametricBetaARChangePointTester:
    def __init__(self, m, N, grid, thresholds, quantile_map, gamma_values=[0, 0.25, 0.4], A=None):
        """
        Initialize the tester with required parameters.
        """
        self.m = m
        self.N = N
        self.grid = grid
        self.thresholds = thresholds
        self.quantile_map = quantile_map
        self.gamma_values = gamma_values
        self.A = A

    def run_test(self, y_list):
        """
        Main function that runs the Ha_test_nonpa logic.
        """
        g = len(self.gamma_values)

        max_SAS = [0] * g
        max_SAS_loc =[0] * g

        y_list_origin = y_list[:self.m]
        c_u_list_orgin = self._get_c_u(y_list_origin)

        k = 1
        while k < (self.N * self.m + 1):
            c_u_list_new = self._get_c_u(y_list[:(k + self.m)])
            Dm_vector = self._Dm(k, c_u_list_new, c_u_list_orgin)

            new_montr = [
                self._rho(k, gamma) ** 2 * np.dot(np.dot(Dm_vector, self.A), Dm_vector)
                for gamma in self.gamma_values
            ]

            max_SAS = [max(a, b) for a, b in zip(max_SAS, new_montr)]
            for i in range(g):
                if (max_SAS[i] >= self.thresholds[1][i]) and (max_SAS_loc[i] == 0):
                    max_SAS_loc[i] = k + self.m

            k += 1

        return max_SAS_loc, max_SAS

    def _Dm(self, k, c_u_list_new, c_u_list_orgin):
        """
        Calculate the test vector Dm(s,k).
        """
        diff_list = [c1 - c2 for c1, c2 in zip(c_u_list_new, c_u_list_orgin)]
        Dm_stat = [(k + self.m) * x / np.sqrt(self.m) for x in diff_list]
        return np.array(Dm_stat)

    def _rho(self, k, gamma):
        """
        Compute the rho scalar used for scaling.
        """
        return (1 + k / self.m) ** (-1) * (k / (self.m + k)) ** (-gamma)

    def _get_c_u(self, X):
        """
        Calculate c_u for each quantile in the quantile map.
        """
        c_u_list = []
        x_quan_pre = -1
        c_u = 0
        for i, (_, x_quan, _) in enumerate(self.quantile_map):
            if x_quan == x_quan_pre:
                c_u_list.append(c_u)
            else:
                x_quan_pre = x_quan
                c_u = np.mean([int(z <= x_quan) for z in X])
                c_u_list.append(c_u)
        return c_u_list