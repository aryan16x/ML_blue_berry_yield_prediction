import numpy as np

x_arr = np.load("x_arr.npy")
y_arr = np.load("y_arr.npy")

x = np.c_[x_arr,y_arr]

print(x)

