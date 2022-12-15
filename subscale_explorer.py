import os
import math
import subprocess
import shutil
import json
import itertools
import functools
import pathlib

import pandas as pd
import numpy as np

import munkres
import generator
import sklearn.cluster


class SubscaleExplorer:

    def __init__(self):
        self.DB = None
        self.labels = None
        self.data_initialized = False

    def generate_database(self, n=10000, d=500, c=10, sub_n=10, sub_d=10, std=0.1):
        self.DB, self.labels = generator.generate_subspacedata(n, d, False, [[sub_n, sub_d, 1, std] for i in range(c)])

        # generates sample6.csv file for the subscale application
        np.savetxt("res/sample6.csv", self.DB, delimiter=",", fmt='%1.8f')

        # generates the corresponding ground_truth.csv file, wich is needed for the scoring.
        # Format of ground_truth.csv is the same as the subscale/dbscan results ([d1,d2,..,dn]-[p1,p2,..,pn])
        with open("res/ground_truth.csv", mode="w") as output_file:
            cluster_counter = 1
            while np.argwhere(self.labels == cluster_counter).size > 0:
                dimensions = list(set(np.argwhere(self.labels == cluster_counter)[:, 1]))
                U = list(set(np.argwhere(self.labels == cluster_counter)[:, 0]))
                cluster_counter += 1
                output_file.write(str(dimensions) + "-" + str(U) + "\n")
        self.data_initialized = True

    def subscale(self, epsilon, minpts, verbose=False):
        if not self.data_initialized:
            raise Exception("Call generate_database() first.")

        # Deletes the /out folder as this is needed between subscale runs
        dirpath = pathlib.Path('out')
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

        SAMPLEFILE = '6'
        GERMAN_FILE_FORMATTING = 'true'
        SPLITTING_SIZE = '16'
        EVENLY_SIZED_SLICES = 'true'
        DBSCAN = 'false'
        # Spawns new processes for SubscaleExtended
        p = subprocess.Popen(['java',
                              '-jar',
                              'SubScaleExtended.jar',
                              SAMPLEFILE,
                              GERMAN_FILE_FORMATTING,
                              str(epsilon),
                              str(minpts),
                              SPLITTING_SIZE,
                              EVENLY_SIZED_SLICES,
                              DBSCAN],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        # Blocks further processing by reading stdout until SubscaleExtended finishes
        for line in p.stdout:
            if verbose:
                print(str(line, "utf-8").replace('\r\n', ''))  # prints logging of SubscaleExtended

    def dbscan(self, eps, adjust_epsilon=False):
        """
        Potential subspaces will be searched for clusters using the dbscan algorithm.
        The input subspaces are in a csv format. The clusters found are also written to a csv file.
        The algorithm is density-based and is able to recognize multiple clusters. Noise points are ignored.
        """
        if not self.data_initialized:
            raise Exception("Call generate_database() first.")

        with open("out/results/found_clusters.csv", mode="w") as output_file, \
                os.scandir("out/results/subspaces") as it:

            # Iterate over the csv files.
            for entry in it:
                if os.path.getsize( entry.path) == 0:
                    # Check for empty file sometimes generated due to bug in SubScaleExtended.jar
                    break
                df = pd.read_csv(entry.path, header=None, delimiter="-")

                if adjust_epsilon:
                    n_dimensions = len(json.loads(df[0][0]))
                    epsilon_adjusted = math.sqrt(eps**2 * n_dimensions)

                # Iterate over all potential subspaces
                for _, row in df.iterrows():
                    points_indexes = json.loads(row[1])
                    dimensions = json.loads(row[0])
                    S = self.DB[points_indexes][:, dimensions]
                    clustering = sklearn.cluster.DBSCAN(eps=epsilon_adjusted if adjust_epsilon else eps).fit(S)

                    # Converting dbscan result in clusters.csv file-format
                    unique_labels = set(clustering.labels_)
                    unique_labels.discard(-1)  # -1 stands for noisy samples and is therefore removed
                    for k in unique_labels:
                        U = np.array(points_indexes)[clustering.labels_ == k]
                        output_file.write(str(dimensions) + "-" + str(U.tolist()) + "\n")

    def score(self, metric="f1"):
        with open("out/results/found_clusters.csv", mode="r") as found_clusters_file, \
                open("res/ground_truth.csv", mode="r") as ground_truth_file:

            # A dataframe read from the csv format is converted in such a way
            # that the dimensions and point_indexes are available as a set in the corresponding columns.
            # Each row is a cluster with column 0 as the dimensions and column 1 as the point_indexes.
            df_found_clusters = pd.read_csv(found_clusters_file, header=None, delimiter="-") \
                .apply(lambda s: s.apply(lambda x: (set(json.loads(x)))))
            df_ground_truth = pd.read_csv(ground_truth_file, header=None, delimiter="-") \
                .apply(lambda s: s.apply(lambda x: (set(json.loads(x)))))

            if metric == "f1":
                return self.f1_measure(df_ground_truth[1], df_found_clusters[1])

            # Clustering with Cartesian product of dimensions and point_indexes as cluster
            # representation. Once for ground_truth, once for found_clusters.
            clusters_cartesian_product_ground = [list(itertools.product(x, y)) for x, y in zip(df_ground_truth[0],
                                                                                               df_ground_truth[1])]
            clusters_cartesian_product_res = [list(itertools.product(x, y)) for x, y in zip(df_found_clusters[0],
                                                                                            df_found_clusters[1])]

            # Entirety of all micro_objects (cartesian products) of the clustering
            # in the form of a set. Once for ground_truth, once for found_clusters.
            micro_objects_ground = set(functools.reduce(lambda a, b: a+b, clusters_cartesian_product_ground))
            micro_objects_res = set(functools.reduce(lambda a, b: a+b, clusters_cartesian_product_res))

            # U is union between all micro_objects from both ground and res
            U = micro_objects_ground.union(micro_objects_res)

            if metric == "rnia":
                # I is intersection between all micro_objects from both ground and res
                I = micro_objects_ground.intersection(micro_objects_res)
                return self.rnia(len(U), len(I))

            if metric == "ce":
                return self.cluster_error(len(U), clusters_cartesian_product_ground, clusters_cartesian_product_res)

    @staticmethod
    def f1_measure(ground, res):
        """
        The basic idea of f1_measure is to find a 1:1 mapping between the ground_truth and found clusters,
        so that the mapped clusters achieve the best possible f1-scores.
        That is, a found cluster should on the one hand have many objects in common with one of the
        hidden clusters, but on the other hand it should contain as few objects as possible
        that are not in this particular hidden cluster (see inner function f1_score).
        The final score results from the mean value of the mapped f1 scores.
        """

        def f1(c_res, c_ground):
            """Harmonic mean of precision and recall between the 2 input clusters"""
            try:
                recall = len(c_res.intersection(c_ground)) / len(c_ground)
                precision = len(c_ground.intersection(c_res)) / len(c_res)
                return (2 * recall * precision) / (recall + precision)
            except ZeroDivisionError:
                return 0

        accumulated_scores = 0
        for c_ground in ground:
            best_score = 0
            for c_res in res:
                best_score = max(f1(c_ground, c_res), best_score)
            accumulated_scores += best_score
        try:
            return accumulated_scores / len(ground)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def rnia(size_union, size_intersection):
        """
        Relative NonIntersecting Area (RNIA)
        Formula: 1 - (size_union - size_intersection)/size_union

        :param size_union: Cardinality of the union between all micro_objects from both ground_truth and found_clusters.
        :param size_intersection: Cardinality of the intersection between all micro_objects from both ground_truth
        and found_clusters.
        """
        return 1 - (size_union - size_intersection)/size_union

    @staticmethod
    def cluster_error(size_union, ground, res):
        """
        The basic idea of cluster_error (CE) is to find a 1:1 mapping between the ground_truth and found clusters.
        For each mapped pair (Cg , Cr ) the cardinality of their intersecting micro_objects is determined.
        Overall, those 1:1 mappings are chosen that result in the highest total sum over all cardinalities.
        This sum is denoted as D_max.
        The formula to calculate cluster_error is: 1 - (size_union - D_max)/size_union
        """

        # M is the confusion matrix between the found_clusters and ground_truth clusters. The values
        # of the individual entries result from the cardinality of the intersection of the respective clusters.
        M = pd.DataFrame(itertools.product(ground,res)).apply(lambda x: len(set(x[0]).intersection(set(x[1]))), axis=1)\
            .to_numpy().reshape(len(ground),len(res))

        # We use the Hungarian method to find a permutation of the cluster labels such
        # that the sum of the diagonal elements of M is maximized.
        indexes = munkres.Munkres().compute(munkres.make_cost_matrix(M))
        D_max = 0
        for row, column in indexes:
            value = M[row][column]
            D_max += value

        return 1 - ((size_union - D_max) / size_union)

