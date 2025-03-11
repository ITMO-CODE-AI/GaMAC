from numba import cuda, types
import numpy as np
import math


class S_Dbw():
    def __init__(self, data, data_cluster, centers_index):
        self.data = data
        self.data_cluster = data_cluster
        self.centers_index = centers_index

        self.k = len(centers_index)

        # Initialize stdev
        self.stdev = 0
        for i in range(self.k):
            std_matrix_i = np.std(data[self.data_cluster == i], axis=0)
            self.stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
        self.stdev = math.sqrt(self.stdev) / self.k

    @cuda.jit(device=True)
    def density(self, density_list=[]):
        density = 0
        centers_index1 = self.centers_index[density_list[0]]
        if len(density_list) == 2:
            centers_index2 = self.centers_index[density_list[1]]
            center_v = (self.data[centers_index1] + self.data[centers_index2]) / 2
        else:
            center_v = self.data[centers_index1]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density

    @cuda.jit(device=True)
    def Dens_bw(self):
        density_list = []
        result = 0
        for i in range(self.k):
            density_list.append(self.density(density_list=[i]))
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                result += self.density([i, j]) / max(density_list[i], density_list[j])
        return result / (self.k * (self.k - 1))

    @cuda.jit(device=True)
    def Scat(self):
        theta_s = np.std(self.data, axis=0)
        theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
        sum_theta_2norm = 0

        for i in range(self.k):
            matrix_data_i = self.data[self.data_cluster == i]
            theta_i = np.std(matrix_data_i, axis=0)
            sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
        return sum_theta_2norm / (theta_s_2norm * self.k)

    @staticmethod
    @cuda.jit(())
    def S_Dbw_result(self):
        return self.Dens_bw() + self.Scat()
