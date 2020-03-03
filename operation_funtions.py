import numpy as np


def logistic(x, derivate=False):
    value = 1 / (1 + np.exp(-x))
    if derivate:
        value *= 1 - value
    return value


def softmax(x, derivate=False):
    value = x / np.sum(x, axis=1)[:, np.newaxis]
    if derivate:
        value *= (np.sum(x, axis=1) - x) / np.sum(x, axis=1)[:, np.newaxis]
    return value


def relu(x, derivate=False):
    y = np.copy(x)
    c = x < 0
    if np.any(c):
        y[c] = 0
        if derivate:
            c = x > 0
            if np.any(c):
                y[c] = 1
    return y


def cross_entropy(y, y_hat, derivate=False):
    ac1 = np.argmax(y, axis=1)
    ac2 = np.argmax(y_hat, axis=1)
    accuracy = ac1 == ac2
    if np.any(accuracy):
        accuracy = np.sum(accuracy)
    else:
        accuracy = 0

    if derivate:
        error = -y/y_hat - (1-y)/(1-y_hat)
    else:
        error = -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)
    return error/error.shape[0], accuracy / ac1.size


if __name__ == '__main__':
    a = np.zeros((2, 10))
    a[0, 0], a[1, 1] = 1, 1
    b = np.random.uniform(0, 1, (2, 10))
    b = b / np.sqrt(np.sum(b, axis=1) ** 2)[:, np.newaxis]
    e, ac = cross_entropy(a, b, derivate=False)
    print(e)
    print(ac)
