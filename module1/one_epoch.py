import numpy as np


def epoch(my_w, my_data, my_target):
    for i in range(my_data.shape[0]):
        p = predict(my_w, my_data[i])
        if p > 0:
            p = 1
        else:
            p = 0
        if p != my_target[i]:
            if p == 0:
                my_w += my_data[i]
            else:
                my_w -= my_data[i]


def predict(my_w, my_data):
    return my_w.dot(my_data)


w = np.array([0.0, 0.0, 0.0])
data = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [1, 0.7, 0.8]])
target = np.array([1.0, 1.0, 0.0])

epoch(w, data, target)
print(w)
