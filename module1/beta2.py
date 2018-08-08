from urllib.request import urlopen
import numpy as np

filename = "https://stepic.org/media/attachments/lesson/16462/boston_houses.csv"
f = urlopen(filename)

data = np.loadtxt(f, skiprows=1, delimiter=",")
nn = data.shape[0]
n = data.shape[1]

xx = data[:, 1:n]
y = data[:, 0]
y = y.reshape((nn, 1))

e = np.ones((nn, 1))
x = np.concatenate((e, xx), axis=1)

st1 = x.transpose().dot(x)
st2 = np.linalg.inv(st1)
st3 = st2.dot(x.transpose())

beta = st3.dot(y)
beta = beta.flatten()
print(" ".join(map(str, beta)))
beta = st3.dot(y)
