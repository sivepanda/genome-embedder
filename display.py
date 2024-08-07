# from tensorflow.keras.models import load_model
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from torch_model import Autoencoder
from gencovvec import get_coverage_vectors
from gencovvec import get_labelled_coverage_vectors
from genemb import get_encoded_vals_torch, get_encoded_vals_tf

# Ensures coverage vectors exist
get_coverage_vectors(new_coverage_vectors=False)

print('\n')
print('Select your graph option')
print('[0] 2-Dimensional Graph')
print('[1] 3-Dimensional MatplotLib Graph')
print('[2] 3-Dimensional Plotly Graph')
option = int(input())
print('\n')

if (option < 0 or option > 2):
    print('Invalid option.\nSelecting Default: "2-Dimensional Graph".')
    option = 0

print('\n')
print('Select your model option')
print('[0] PyTorch (ONLY SUPPORTS CUDA)')
print('[1] TensorFlow')
model_option = int(input())
print('\n')

if model_option == 1:
    # Gets encoded values from the PyTorch model
    encoded_vals = get_encoded_vals_torch()
else:
    # Gets encoded values from TensorFlow model if it was selected or if an invalid option was selected
    if model_option > 1:
        print('Invalid option.\nSelecting Default: TensorFlow')
    encoded_vals = get_encoded_vals_tf()

# 2 Dimensional graph
if option == 0:
    e_v = []
    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    labels = list( encoded_vals.keys() )

    tsne = TSNE(n_components = 2)
    embeddings_2d = tsne.fit_transform( e_v )
     
    plt.scatter( embeddings_2d[:, 0], embeddings_2d[:, 1] )
    for i, label in enumerate( labels ):
        plt.annotate( label, ( embeddings_2d[i, 0], embeddings_2d[i, 1] ) )
    plt.savefig(os.path.join(os.getcwd(), 'results', f'fig.png'))


# 3 Dimensional plot with Matplotlib
if option == 1:
    e_v = []
    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    labels = list( encoded_vals.keys() )

    tsne = TSNE(n_components = 3)
    embeddings_3d = tsne.fit_transform( e_v )
     
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter( embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2] )
    for i, label in enumerate( labels ):
        ax.text( embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2] , label )

    plt.show()

# 3 Dimensional plot with Plotly
if option == 2:
    import pandas as pd
    import plotly.express as px
    e_v = []
    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    labels = list( encoded_vals.keys() )
    print('here')

    tsne = TSNE(n_components = 3)
    embeddings_3d = tsne.fit_transform( e_v )
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df['label'] = labels

    # Plot with Plotly
    fig = px.scatter_3d(df, x='x', y='y', z='z', text='label')
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers+text'))
    fig.write_html(os.path.join(os.getcwd(), 'results', 'fig.html'))
