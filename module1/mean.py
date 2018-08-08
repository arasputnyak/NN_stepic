from urllib.request import urlopen
import numpy as np

filename = "https://stepic.org/media/attachments/lesson/16462/boston_houses.csv"
f = urlopen(filename)
sbux = np.loadtxt(f, skiprows=1, delimiter=",")
print(sbux.mean(axis=0))
