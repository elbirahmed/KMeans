import pandas as pd
import numpy as np
from copy import copy
from copy import deepcopy


class KMeans:

    def __init__(self, df, nb_clusters, max_iter=300, nb_init=10, calc_d=False):
        self.df = copy(df)
        self.nb_clusters = nb_clusters
        self.nb_init = nb_init
        self.max_iter = max_iter
        self.calc_d = calc_d
        self.inertia = np.inf
        self.nb_iter_run = 0
        self.centroids = {}
        if self.calc_d:
            self.__df_distance = pd.DataFrame(index=df.index, columns=df.index)

    def __compute_distance(self):
        for i in self.df.index:
            for j in self.df.index:
                val = np.sqrt(sum([(self.df.iat[i, 0] - self.df.iat[j, 0]) ** 2, (self.df.iat[i, 1] - self.df.iat[j, 1]) ** 2]))
                self.__df_distance.iat[i, j] = val

    def __init_iteration(self):
        self.nb_iter_run = 0
        self.centroids = {i: val for i, val in zip(range(self.nb_clusters), self.df.sample(n=self.nb_clusters).values)}

    def __affectation(self):

        for k, v in self.centroids.items():
            self.df['distance_from_{}'.format(k)] = np.sqrt((self.df.x - v[0]) ** 2 + (self.df.y - v[1]) ** 2)
        dist_col_names = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
        self.df['closest'] = self.df.loc[:, dist_col_names].idxmin(axis=1)
        self.df['closest'] = self.df.closest.map(lambda x: x[-1])

    def __compute_centroids(self):
        for k in self.centroids.keys():
            new_x = np.mean(self.df[self.df.closest == str(k)].x)
            new_y = np.mean(self.df[self.df.closest == str(k)].y)
            self.centroids[k] = np.array([new_x, new_y])

    def __compute_inertia(self):
        val = 0
        for k in self.centroids.keys():
            val += np.sum(self.df[self.df.closest == str(k)]['distance_from_{}'.format(k)] ** 2)
        self.inertia = val

    def _compute(self):

        b_compute = True
        self.nb_iter_run = 0

        while b_compute and self.nb_iter_run <= self.max_iter:
            self.__affectation()
            old_centroids = deepcopy(self.centroids)
            self.__compute_centroids()
            b_compute = not all(np.array_equal(self.centroids[i], old_centroids[i]) for i in old_centroids.keys())
            self.nb_iter_run += 1
        self.__compute_inertia()

    def fit(self):
        if self.calc_d:
            self.__compute_distance()
        tmp_inertia = np.inf
        for i in range(self.nb_init):
            self.__init_iteration()
            self._compute()
            if tmp_inertia > self.inertia:
                tmp_inertia = self.inertia
                tmp_centroids = deepcopy(self.centroids)
                tmp_df_res = deepcopy(self.df.closest)
                tmp_nb_iter_run = self.nb_iter_run
        self.inertia = tmp_inertia
        self.centroids = deepcopy(tmp_centroids)
        self.df.closest = tmp_df_res
        self.nb_iter_run = tmp_nb_iter_run


if __name__ == "__main__":

    df = pd.DataFrame({'x': [1, 3, 3, 3, 3, 5], 'y': [2, 1, 2, 3, 4, 2]})
    k_means = KMeans(df, 2, nb_init=20, calc_d=False)
    k_means.fit()
    print(k_means.df)
    print(k_means.centroids)
    print(k_means.nb_iter_run)
    print(k_means.inertia)


