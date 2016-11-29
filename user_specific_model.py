"""
    user_specific_model

    Definition of the UserProfile model.
    The class UserSpecializedEstimator implements the sklearn classifier interface
    - usercol is the column with the user ID
    - threshold is the minimum number of pickups necessary to have a specialized model
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone

__author__ = 'Antonio Carta'


class UserSpecializedEstimator(BaseEstimator):
    def __init__(self, model, usercol, threshold):
        self.model = model
        self.usercol = usercol
        self.threshold = threshold

        self.standardModel = clone(self.model)
        self.specializedModels = dict()

    def fit(self, X, y):
        gb = X.groupby(self.usercol).size()
        stdusr = gb[gb < self.threshold]
        a = X[self.usercol].isin(stdusr.index)

        if sum(a) > 0:
            self.standardModel.fit(X[a], y[a])

        regusr = gb[gb >= self.threshold]
        for (k, v) in regusr.iteritems():
            # print("\t" + str(k) + ": " + str(v))
            model = clone(self.model)
            itrain = X[self.usercol] == k
            try:
                model.fit(X[itrain], y[itrain])
                self.specializedModels[k] = model
            except ValueError as e:
                print("Error during training: ValueError (" + str(k) + ")")

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(0, len(X)):
            el = X.iloc[i]
            if el[self.usercol] in self.specializedModels:
                el_pred = self.specializedModels[el[self.usercol]].predict(el.transpose())
                y_pred[i] = el_pred
            else:
                el_pred = self.standardModel.predict(el.transpose())
                y_pred[i] = el_pred
        return y_pred


