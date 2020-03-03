from operation_funtions import *
import numpy as np


class NN:

    def __init__(self,_deep_layer_nodes,_out_layer_nodes, _filters,_lr=0.5):
        nb_filt = 9
        for f in _filters:
            nb_filt *= f
        # Input lauyer
        self.n1 = None
        self.w1 = np.random.uniform(-0.5, 0.5, (nb_filt, _deep_layer_nodes))
        self.b1 = np.random.uniform(0, 0.1, 1)
        # deep layers
        self.n2 = None
        self.w2 = w2 = np.random.uniform(-0.5, 0.5, (_deep_layer_nodes, _deep_layer_nodes))
        self.b2 = np.random.uniform(0, 0.1, 1)
        self.n3 = None
        # output layer
        self.b3 = np.random.uniform(0, 0.1, 1)
        self.w3 = np.random.uniform(0, 0.5, (_deep_layer_nodes, _out_layer_nodes))
        self.lr = _lr

    def forward_pass(self, _inpu):
        im_out = np.reshape(_inpu, (_inpu.shape[0], _inpu.shape[1], _inpu.shape[2] * _inpu.shape[3]))
        self.n1 = np.reshape(im_out, (im_out.shape[0], im_out.shape[1] * im_out.shape[2]))
        self.n2 = logistic(self.n1 @ self.w1 + self.b1)
        self.n3 = logistic(self.n2 @ self.w2 + self.b2)
        z = self.n3 @ self.w3 + self.b3  # used in variable out and for computing the derivate in error
        out = softmax(z)
        return out

    def backward_pass(self, error):
        self.w3 -= self.lr * self.n3.T @ error
        self.b3 -= self.lr * np.sum(error)
        error = error @ self.w3.T * logistic(self.n3, derivate=True)
        self.w2 -= self.lr * self.n2.T @ error
        self.b2 -= self.lr * np.sum(error)
        error = error @ self.w2.T * logistic(self.n2, derivate=True)
        self.w1 -= self.lr * self.n1.T @ error
        self.b1 -= self.lr * np.sum(error)
        error = error @ self.w1.T * self.n1
        return error
