from tensorflow.keras.models import load_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from gencovvec import get_labelled_coverage_vectors

dim = 4 # define the number of dimensions

np.set_printoptions(threshold=np.inf)
lbc = get_labelled_coverage_vectors()
encoder = load_model('./encoder.keras')
encoded_vals = {}
for key, value in lbc.items():
    print(key)
    encoded_vals[ key ] = encoder.predict( np.array( [value] ) )
    print( encoded_vals[ key ] )


if dim == 2:
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
    plt.show()



if dim == 3:
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

if dim == 4:
    import pandas as pd
    import plotly.express as px
    e_v = []
    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    labels = list( encoded_vals.keys() )

    tsne = TSNE(n_components = 3)
    embeddings_3d = tsne.fit_transform( e_v )
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df['label'] = labels

    # Plot with Plotly
    fig = px.scatter_3d(df, x='x', y='y', z='z', text='label')
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers+text'))
    fig.show()
