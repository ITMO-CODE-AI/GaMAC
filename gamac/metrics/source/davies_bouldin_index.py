from numba import cuda, types
from ..utils.funcs import euclidean_distances_numba


class DaviesBouldinIndex:

    def __init__(self, ):
        pass

    @staticmethod
    @cuda.jit(device=True)
    def s(i, x, labels, clusters):
        norm_c = len(clusters)
        s = 0
        for c in clusters:
            s += euclidean_distances_numba(c, clusters[i])
        return s / norm_c

    @cuda.jit(device=True)
    def Rij(self, i, j, x, labels, clusters, nc):
        Rij = 0.0
        try:
            d = euclidean_distances_numba(clusters[i], clusters[j])
            Rij = (self.s(i, x, labels, clusters) + self.s(j, x, labels, clusters)) / d
        except:
            Rij = 0
        return Rij
    @cuda.jit(device=True)
    def R(self, i, x, labels, clusters, nc):
        list_r = []
        for i in range(nc):
            for j in range(nc):
                if(i != j):
                    temp = self.Rij(i, j, x, labels, clusters, nc)
                    list_r.append(temp)

        return max(list_r)

    @cuda.jit(())
    def DB_index(self, x, labels, clusters, nc):
        sigma_R = 0.0
        for i in range(nc):
            sigma_R = sigma_R + self.R(i, x, labels, clusters, nc)

        DB_index = float(sigma_R) / float(nc)
        return DB_index
