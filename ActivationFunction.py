import numpy as np


class ActivationFunction:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def sign(z):
        return np.sign(z)

    def sigmoid_derivative(self, da, z):
        sig = self.sigmoid(z)
        return da * sig * (1 - sig)

    def tanh_derivative(self, da, z):
        tanh = self.tanh(z)
        return da*(1-tanh**2)

    @staticmethod
    def sign_derivative(self, da, z):
        return np.zeros(z.shape[0], da.shape[0])
