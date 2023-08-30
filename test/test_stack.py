import numpy as np

a = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# a = np.array([1, 2, 3, 4])
# b = np.array([2, 3, 4, 5])
# c = np.array([6, 7, 8, 9])

my_stack = np.stack([a, a, a], axis=1)
print(my_stack)
