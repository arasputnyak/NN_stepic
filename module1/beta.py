import numpy as np

e = np.ones((3, 1))
x = np.array([60, 50, 75]).reshape((3, 1))
y = np.array([10, 7, 12])

xx = np.concatenate((e, x), axis=1)

st1 = xx.transpose().dot(xx)
st2 = np.linalg.inv(st1)
st3 = st2.dot(xx.transpose())
beta = st3.dot(y)

print(beta)