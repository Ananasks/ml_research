"""Compatibility adapters preserving original class labels for third-party models."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class EncodedClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,estimator): self.estimator=estimator

    def fit(self,X,y):
        self.encoder_=LabelEncoder().fit(y)
        self.classes_=self.encoder_.classes_
        self.estimator_=clone(self.estimator).fit(X,self.encoder_.transform(y))
        self.n_features_in_=X.shape[1]
        return self

    def predict(self,X):
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X),dtype=int))

    def predict_proba(self,X): return self.estimator_.predict_proba(X)
