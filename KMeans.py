import pandas as pd
import numpy as np
from copy import copy
from copy import deepcopy


class KMeans:

    def __init__(self, df, nb_clusters):

        self._df = copy(df)
        self._nb_clusters = nb_clusters
        self._centroids = {i: val for i, val in zip(range(nb_clusters), self._df.sample(n=nb_clusters).values)}

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

    def fit(self):

        b_compute = True

        while b_compute:

            self.__affectation()
            old_centroids = deepcopy(self._centroids)
            self.__compute_centroids()
            b_compute = old_centroids == self._centroids


if __name__ == "__main__":

    df = pd.DataFrame({'x': [1, 2, 8, 10], 'y': [3, 2, 25, 14]})
    k_means = KMeans(df, 2)
    k_means.fit()