import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler


class Estimator:
    def __init__(self, model, scaler, threshold):
        """
        Error estimator using a neural network model.

        Parameters
        ----------
        model : tf.keras.Model
        scaler : StandardScaler
        threshold : float
        """
        self.scaler = scaler
        self.model = model
        self.threshold = threshold

    def is_error_large(self, state):
        """
        If the error is estimated to be larger than threshold, outputs True.

        Parameters
        ----------
        state : np.ndarray

        Returns
        -------
        bool
        """
        error = self.model(self.scaler.transform([state])).numpy()
        return bool(np.argmax(error))
