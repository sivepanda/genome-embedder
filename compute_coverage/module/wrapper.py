import ctypes
import os

# Load the shared library
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "gcc.so"))

# Specify the argument and return types of the C function
_lib.add.argtypes = (ctypes.c_int, ctypes.c_int)
_lib.add.restype = ctypes.c_int

def add(a, b):
    return _lib.add(a, b)
