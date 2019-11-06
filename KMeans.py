import pandas as pd
import numpy as np
from copy import copy
from copy import deepcopy


class KMeans:

    def __init__(self, df, nb_clusters, max_iter=300):

        self._df = copy(df)
        self._nb_clusters = nb_clusters
        self._centroids = {i: val for i, val in zip(range(nb_clusters), self._df.sample(n=nb_clusters).values)}
        self._inertia = 0
        self._max_iter = max_iter
        self._nb_iter_run = 0


    def __affectation(self):

        for k, v in self._centroids.items():
            self._df['distance_from_{}'.format(k)] = np.sqrt((df.x - v[0]) ** 2 + (df.y - v[1]) ** 2)

        dist_col_names = ['distance_from_{}'.format(i) for i in self._centroids.keys()]
        self._df['closest'] = self._df.loc[:, dist_col_names].idxmin(axis=1)
        self._df['closest'] = self._df.closest.map(lambda x: x[-1])

    def __compute_centroids(self):

        for k in self._centroids.keys():
            new_x = np.mean(self._df[self._df.closest == str(k)].x)
            new_y = np.mean(self._df[self._df.closest == str(k)].y)
            self._centroids[k] = np.array([new_x, new_y])

    def __compute_inertia(self):
        for k in self._centroids.keys():
            self._inertia = self._inertia + np.sum(self._df[self._df.closest == str(k)]['distance_from_{}'.format(k)] ** 2)

    def fit(self):

        b_compute = True
        self._nb_iter_run = 0

        while b_compute and self._nb_iter_run <= self._max_iter:
            self.__affectation()
            old_centroids = deepcopy(self._centroids)
            self.__compute_centroids()
            b_compute = not all(np.array_equal(self._centroids[i], old_centroids[i]) for i in old_centroids.keys())
            self._nb_iter_run += 1
        self.__compute_inertia()


if __name__ == "__main__":

    df = pd.DataFrame({'x': [1, 3, 3, 3, 3, 5], 'y': [2, 1, 2, 3, 4, 2]})
    k_means = KMeans(df, 2)
    k_means.fit()
    print(k_means._df)
    print(k_means._centroids)
    print(k_means._nb_iter_run)
    print(k_means._inertia)
