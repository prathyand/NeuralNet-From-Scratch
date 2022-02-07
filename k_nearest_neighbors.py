# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Prathmesh Narayanrao Deshmukh] -- [pdeshmuk]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X=X
        self._y=y
#         raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        
        result=np.array([])
        for indq,valq in enumerate(X):
        #     if(indq==1):
        #         break
        #     print("test example ",valq)
            kn=[]
            for ind,val in enumerate(self._X):
                kn.append((self._distance(val,valq),ind))
#                 if(ind==15):
#                     break
            kn=sorted(kn,key=lambda t:t[0])
            kni=[t[1] for t in kn[:self.n_neighbors]]
            if(self.weights in ['uniform']):
                result=np.append(result,np.bincount(np.array([self._y[ww] for ww in kni])).argmax())
            else:
                dist=[1/(1+t[0]) for t in kn[:self.n_neighbors]]
                dic={}
                for oo in range(len(dist)):
                    if(self._y[kni[oo]] not in dic):
                        dic[self._y[kni[oo]]]=dist[oo]
                    else:
                        dic[self._y[kni[oo]]]+=dist[oo]
                result=np.append(result,max(dic,key=dic.get))
                
#             print(np.bincount(np.array([_y[ww] for ww in kni])).argmax())
            
        return result
#         print(result.size)
#         raise NotImplementedError('This function must be implemented by the student.')
