import pandas as pd
import numpy as np
from copy import copy
from copy import deepcopy
import inspect


class KMeans:

    def __init__(self, df, nb_clusters, max_iter=300, nb_init=10, init='kmeans_pp', centroids={}):
        self.df = copy(df)
        self.nb_clusters = nb_clusters
        self.nb_init = nb_init
        self.max_iter = max_iter
        self.inertia = np.inf
        self.nb_cordinates = self.df.shape[1]
        self.nb_iter_run = 0
        self.init_centroids = centroids
        self.centroids = {}
        self.__build_strategy_dict()
        self.init = self.init_strategies["_KMeans__init_iteration_strategy_{}".format(init)]
        # self.calc_d = calc_d
        # if self.calc_d:
        #   self.__df_distance = pd.DataFrame(index=df.index, columns=df.index)

    def __compute_distance(self):
        for i in self.df.index:
            for j in self.df.index:
                val = np.sqrt(sum([(self.df.iat[i, 0] - self.df.iat[j, 0]) ** 2, (self.df.iat[i, 1] - self.df.iat[j, 1]) ** 2]))
                self.__df_distance.iat[i, j] = val

    def __init_iteration_strategy_kmeans_pp(self):
        self.centroids[0] = self.df.sample(n=1).values[0]
        indexes = [i for i in range(self.nb_cordinates)]
        for key in range(1, self.nb_clusters):
            for k, v in self.centroids.items():
                self.df['distance_from_{}'.format(k)] = np.sqrt(
                    sum((self.df.iloc[:, i] - v[i]) ** 2 for i in range(self.nb_cordinates)))
            dist_col_names = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
            self.df['closest'] = self.df.loc[:, dist_col_names].min(axis=1)
            fact = 1/sum(self.df['closest']**2)
            s = self.df['closest'].map(lambda x: (x**2) * fact)
            self.centroids[key] = self.df.iloc[:, indexes].sample(n=1, weights=s).values[0]
        self.df = self.df.iloc[:, indexes]

    def __init_iteration_strategy_random(self):
        self.centroids = {i: val for i, val in zip(range(self.nb_clusters), self.df.sample(n=self.nb_clusters).values)}

    def __init_iteration_strategy_assigned(self):
        self.centroids = deepcopy(self.init_centroids)

    def __init_iteration(self):
        self.nb_iter_run = 0
        self.init(self)


    def __affectation(self):

        for k, v in self.centroids.items():
            self.df['distance_from_{}'.format(k)] = np.sqrt(sum((self.df.iloc[:, i] - v[i]) ** 2 for i in range(self.nb_cordinates)))
        dist_col_names = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
        self.df['closest'] = self.df.loc[:, dist_col_names].idxmin(axis=1)
        self.df['closest'] = self.df.closest.map(lambda x: x[-1])

    def __compute_centroids(self):
        for k in self.centroids.keys():
            new = [np.mean(self.df[self.df.closest == str(k)].iloc[:, i]) for i in range(self.nb_cordinates)]
            self.centroids[k] = np.array(new)

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
        # if self.calc_d:
        #   self.__compute_distance()
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

    @classmethod
    def __build_strategy_dict(cls):
        cls.init_strategies = {func_name: func for func_name, func
         in inspect.getmembers(cls, inspect.isfunction) if func_name.startswith("_KMeans__init_iteration_strategy_")}


if __name__ == "__main__":

    df = pd.DataFrame({'x': [1, 3, 3, 3, 3, 5, 6, 8, 10], 'y': [2, 1, 2, 3, 4, 2, 6, 8, 10]})
    k_means = KMeans(df, 2, nb_init=20, init="assigned", centroids={'0': np.array([1, 2]), '1': np.array([2.5, 3.8])})
    k_means.fit()
    print(k_means.df)
    print(k_means.centroids)
    print(k_means.nb_iter_run)
    print(k_means.inertia)


