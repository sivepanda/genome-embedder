import os
from ctypes import *
so_file = os.path.join(os.getcwd(), "gcc.so")


my_functions = CDLL(so_file)

print(type(my_functions))

print(my_functions.square(10))

print(my_functions.square(8))
