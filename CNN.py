from operation_funtions import *
import numpy as np
import matplotlib.pyplot as plt
import time


class CNN:

    def __init__(self, _num_filt, _lr=0.5):
        self.filter = np.random.uniform(-0.1, 0.1, (_num_filt, 3, 3))
        self.bf = np.random.uniform(0, 0.05, _num_filt)
        self.lr = _lr
        self.convolved_image = None
        self.pooled_image = None

    def forward_pass(self, _input, _pool=True):
        self.convolution(_input)
        if _pool:
            self.pool()

    def backward_pass(self, _error, _previous, _pool=True):
        _new_error = _error
        if _pool:
            _new_error = self.back_pool(dy=_new_error)
        else:
            _new_error = np.reshape(_new_error,newshape=self.convolved_image.shape)
        _new_error = self.back_conv2(dYdX=_new_error, previous=_previous)
        return _new_error

    def convolution(self, input_image):
        # Reshape filters and biases to avoid looping on filters
        filts = np.repeat(self.filter, repeats=input_image.shape[1], axis=0)
        bfs = np.repeat(self.bf, repeats=input_image.shape[1], axis=0)
        # Reshaped input image to avoid loop and result image
        self.convolved_image = np.zeros((input_image.shape[0], self.filter.shape[0] * input_image.shape[1],
                                         input_image.shape[2] - 2, input_image.shape[3] - 2))
        new_input = np.zeros((input_image.shape[0], self.filter.shape[0] * input_image.shape[1], input_image.shape[2],
                              input_image.shape[3]))
        for i in range(self.filter.shape[0]):
            if input_image.shape[1] == 1:
                new_input[:, i * input_image.shape[1]:(i + 1) * input_image.shape[1] + 1, :, :] = input_image
            else:
                new_input[:, i * input_image.shape[1]:(i + 1) * input_image.shape[1], :, :] = input_image

        # Convolution
        for i in range(self.convolved_image.shape[2]):
            for j in range(self.convolved_image.shape[3]):
                convol = new_input[:, :, i:i + 3, j:j + 3] * filts
                self.convolved_image[:, :, i, j] = relu(np.sum(convol, axis=(2, 3)) + bfs)

    def back_conv2(self, dYdX, previous):
        # resize filter for error propagation
        filtr = np.repeat(self.filter, int(self.convolved_image.shape[1] / self.filter.shape[0]), axis=0)
        #
        prev = np.repeat(previous, self.filter.shape[0], axis=1)
        #
        dX = np.zeros(previous.shape)
        # For every i and j pixel of the current image
        for i in range(dYdX.shape[2]):
            for j in range(dYdX.shape[3]):
                # Error propagation weights
                err_kij = dYdX[:, :, i, j] * relu(self.convolved_image[:, :, i, j], derivate=True)
                err_kij = err_kij[:, :, np.newaxis, np.newaxis]

                # Filter weights
                dW0 = np.sum(prev[:, :, i:+3 + i, j:+3 + j] * err_kij, axis=0)
                dW0 = np.reshape(dW0,
                                 (self.filter.shape[0], previous.shape[1], self.filter.shape[1], self.filter.shape[2]))
                self.filter -= self.lr * np.sum(dW0, axis=1)

                # Bias weights
                dfb = np.sum(err_kij, axis=0)
                dfb = np.reshape(dfb, (self.filter.shape[0], previous.shape[1]))
                self.bf -= self.lr * np.sum(dfb, axis=1)

                # Error propagation next layer
                dX0 = err_kij[:] * filtr
                dX0 = np.reshape(dX0[:, np.newaxis, :, :, :], (
                    previous.shape[0], self.filter.shape[0], previous.shape[1], self.filter.shape[1],
                    self.filter.shape[2]))
                dX[:, :, i:i + 3, j:j + 3] -= self.lr * np.sum(dX0, axis=1)
        return dX

    def pool(self):
        new_image = np.zeros((1, self.convolved_image.shape[0] * self.convolved_image.shape[1],
                              int(self.convolved_image.shape[2] / 2), int(self.convolved_image.shape[3] / 2)))
        inp = np.resize(self.convolved_image, (1, self.convolved_image.shape[0] * self.convolved_image.shape[1],
                                               self.convolved_image.shape[2], self.convolved_image.shape[3]))
        for i in range(int(self.convolved_image.shape[2] / 2)):
            for j in range(int(self.convolved_image.shape[3] / 2)):
                new_image[:, :, i, j] = np.max(inp[:, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)], axis=(2, 3))
        self.pooled_image = np.resize(new_image, (self.convolved_image.shape[0], self.convolved_image.shape[1],
                                                  int(self.convolved_image.shape[2] / 2),
                                                  int(self.convolved_image.shape[3] / 2)))

    def back_pool(self, dy):
        dY=dy
        #dY = np.reshape(dy, self.pooled_image.shape)
        curr_shape = (1, self.pooled_image.shape[0] * self.pooled_image.shape[1], self.pooled_image.shape[2],
                      self.pooled_image.shape[3])
        new_shape = (1, self.convolved_image.shape[0] * self.convolved_image.shape[1], self.convolved_image.shape[2],
                     self.convolved_image.shape[3])
        curr = np.reshape(self.pooled_image, curr_shape)
        prev = np.reshape(self.convolved_image, new_shape)
        dX = np.zeros(new_shape)
        dY = np.reshape(dY, curr.shape)
        for i in range(curr.shape[2]):
            for j in range(curr.shape[3]):
                currre = curr[:, :, i, j]
                comp = currre[:, :, np.newaxis, np.newaxis] == prev[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3]
                if np.any(comp):
                    dYY = dY[:, :, i, j]
                    dYY = dYY[:, :, np.newaxis, np.newaxis]
                    dYY = np.repeat(dYY, comp.shape[2], axis=(2))
                    dYY = np.repeat(dYY, comp.shape[3], axis=(3))
                    dX[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3] = dYY * comp
        return np.reshape(dX, self.convolved_image.shape)
