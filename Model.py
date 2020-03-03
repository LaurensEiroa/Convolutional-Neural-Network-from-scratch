from CNN import CNN
from NN import NN
from operation_funtions import *
import matplotlib.pyplot as plt


class Model:
    def __init__(self, _filters):
        self.cnn = []
        for f in _filters:
            self.cnn.append(CNN(f))
        self.nn = NN(64, 10, _filters)

    def run(self, input_images, input_labels=None, training=False):
        # Forward pass
        # out = input_images
        for i, cnn_layer in enumerate(self.cnn):
            if i == 0:
                cnn_layer.forward_pass(input_images)
            elif i<len(self.cnn)-1:
                cnn_layer.forward_pass(self.cnn[i - 1].pooled_image)
            else:
                cnn_layer.forward_pass(self.cnn[i - 1].pooled_image,_pool=False)
            # TODO error is here ? the secon cnn layer is ==0
        out = self.nn.forward_pass(self.cnn[-1].convolved_image)
        error, accuracy = cross_entropy(input_labels, out)

        if training:
            err, _ = cross_entropy(input_labels, out, derivate=True)  # TODO
            err = self.nn.backward_pass(error=err)
            for i_layer in reversed(range(len(self.cnn))):
                if i_layer == len(self.cnn) - 1:
                    err = self.cnn[i_layer].backward_pass(err, self.cnn[-1].convolved_image, _pool=False)
                elif i_layer > 0:
                    err = self.cnn[i_layer].backward_pass(err, self.cnn[i_layer - 1].pooled_image)
                else:
                    err = self.cnn[i_layer].backward_pass(err, input_images)
        return error, accuracy
