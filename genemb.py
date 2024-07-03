from tensorflow.keras.models import load_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from gencovvec import get_labelled_coverage_vectors

def get_encoded_vals(model_file='./models/encoder.keras', verbose_output=False):
    np.set_printoptions(threshold=np.inf)
    lbc = get_labelled_coverage_vectors()
    encoder = load_model(model_file)
    encoded_vals = {}
    for key, value in lbc.items():
        print(key)
        encoded_vals[ key ] = encoder.predict( np.array( [value] ) )
        if verbose_output:
            print( encoded_vals[ key ] )
    return encoded_vals
