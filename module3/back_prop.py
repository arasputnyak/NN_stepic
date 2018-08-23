import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def get_error(deltas, sums, weights, func=sigmoid_prime):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    # here goes your code
    return np.mean(deltas.dot(weights) * func(sums), axis=0)


def sum_func(array, weights):
    return weights.dot(array.T).T


def new_act(array):
    new_array = array > 0
    return array * new_array


def new_act_dev(array):
    return array > 0


# first task

# inpt = np.array([[0, 1, 2]])
#
# w2 = np.array([[2, 2, 2],
#                [2, 2, 2]])
#
# w3 = np.array([[1, 1]])
#
# sm2 = sum_func(inpt, w2)
# a2 = sigmoid(sm2)
#
# sm3 = sum_func(a2, w3)
# a3 = sigmoid(sm3)
#
# d3 = (a3 - 1) * sigmoid_prime(sm3)
# d2 = get_error(d3, sm2, w3)
# print(inpt[0][2] * d2)

# second task

# inpt = np.array([[0, 1, 2]])
#
# w2 = np.array([[8.0, 7.0, 8.0],
#                [10.0, 10.0, 9.0]])
#
# w3 = np.array([[10.0, 9.0]])
#
# sm2 = sum_func(inpt, w2)
# a2 = sigmoid(sm2)
#
# sm3 = sum_func(a2, w3)
# a3 = sigmoid(sm3)
#
# d3 = (a3 - 1) * sigmoid_prime(sm3)
# d2 = get_error(d3, sm2, w3)
# print(inpt[0][2] * d2)

# third task

# inpt = np.array([[15.0, 5.0, 15.0]])
#
# w2 = np.array([[0.2, 0.9, 0.6],
#                [0.2, 0.3, 0.7]])
#
# w3 = np.array([[0.2, 0.5]])
#
# sm2 = sum_func(inpt, w2)
# a2 = sigmoid(sm2)
#
# sm3 = sum_func(a2, w3)
# a3 = sigmoid(sm3)
#
# d3 = (a3 - 1) * sigmoid_prime(sm3)
# d2 = get_error(d3, sm2, w3)
# print(inpt[0][2] * d2)

# forth task

inpt = np.array([[0.0, 1.0, 1.0]])

w2 = np.array([[0.7, 0.2, 0.7],
               [0.8, 0.3, 0.6]])

w3 = np.array([[0.2, 0.4]])

sm2 = sum_func(inpt, w2)
a2 = np.array([[new_act(sm2[0][0]), sigmoid(sm2[0][1])]])

sm3 = sum_func(a2, w3)
a3 = sigmoid(sm3)
d3 = (a3 - 1) * sigmoid_prime(sm3)
d2 = get_error(d3, sm2, w3)

d2_0 = get_error(d3, sm2[0][0], w3[0][0], func=new_act_dev)
d2_1 = get_error(d3, sm2[0][1], w3[0][1])
d2 = np.array([[d2_0, d2_1]])
print(inpt[0][2] * d2)
