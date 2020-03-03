import numpy as np
import os
from Model import Model


def load():
    data_sets = ['training_digits_images_mnist.npy', 'training_digits_labels_mnist.npy', 'test_digits_images_mnist.npy',
                 'test_digits_labels_mnist.npy']
    """, 'training_fashion_images_mnist.npy',
    'training_fashion_labels_mnist.npy', 'test_fashion_images_mnist.npy', 'test_fashion_labels_mnist.npy']
    """
    path = 'C:\\Users\\Laurens.Eiroa\\Documents\\ML_dataSets'
    print(os.path.join(path, data_sets[3]))

    x_Dtraining = np.load(os.path.join(path, data_sets[0]))
    x_Dtest = np.load(os.path.join(path, data_sets[2]))
    #y_Dtraining = np.load(os.path.join(path, data_sets[1]))
    #y_Dtest = np.load(os.path.join(path, data_sets[3]))
    """
    x_Ftraining = np.load(os.path.join(path, data_sets[4]))
    x_Ftest = np.load(os.path.join(path, data_sets[6]))
    """
    y_Dtraining, y_Dtest = np.zeros((x_Dtraining.shape[0], 10)), np.zeros((x_Dtest.shape[0], 10))
    """
    y_Ftraining, y_Ftest = np.zeros((x_Ftraining.shape[0], 10)), np.zeros((x_Ftest.shape[0], 10))
    """
    for i, label in enumerate(np.load(os.path.join(path, data_sets[1]))):
        y_Dtraining[i, label] = 1
    for i, label in enumerate(np.load(os.path.join(path, data_sets[3]))):
        y_Dtest[i, label] = 1
    """
    for i, label in enumerate(np.load(os.path.join(path, data_sets[5]))):
        y_Ftraining[i, label] = 1
    for i, label in enumerate(np.load(os.path.join(path, data_sets[7]))):
        y_Ftest[i, label] = 1
    """
    return x_Dtraining / 255 - 0.5, y_Dtraining, x_Dtest/255-0.5, y_Dtest


def deploy(brain,im,imt,y_t,y_test,num_imag=100):
    import matplotlib.pyplot as plt
    print(im.shape)
    for epoch in range(1):
        loss, acuracy = brain.run(im, y_t[0:num_imag], training=True)
        print("epoch:", epoch, "Training  -->  loss:", round(np.sum(loss)/loss.size, 4), "acuracy", round(acuracy*100,4), "%")
        loss, acuracy = brain.run(imt, y_test[0:num_imag])
        print("epoch:", epoch, "Test  ------>  loss:", round(np.sum(loss)/loss.size, 4), "acuracy", round(acuracy*100,4),"%")
    return brain,im,imt,y_t,y_test


if __name__ == '__main__':
    num_imag = 100
    x_t, y_t, x_test, y_test = load()
    # training images
    im = np.reshape(x_t[:, np.newaxis, :, :], (x_t.shape[0], 1, x_t.shape[1], x_t.shape[2]))[0:num_imag]
    # test images
    imt = np.reshape(x_test[:, np.newaxis, :, :], (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))[0:num_imag]
    filters = [2, 4, 4]
    error_vs_epoch = []

    brain = Model(filters)

    print("start")
    brain, im, imt, y_t, y_test = deploy(brain, im, imt, y_t, y_test, num_imag)
