from tensorflow.keras.models import load_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from gencovvec import get_labelled_coverage_vectors
from genemb import get_encoded_vals 
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

# get_coverage_vectors(new_coverage_vectors=True)

options = ['n-nearest-cosine', '3-dimensional-graph-matplot', '3-dimensional-graph-plotly']
option = options[0]

encoded_vals = get_encoded_vals()

if option == options[0]:
    e_v = []
    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    embedding_keys = list( encoded_vals.keys() )
    similarity_matrix = cosine_similarity(e_v)

    n = 5
    top_n_similar_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:n+1]  # Exclude self by starting at index 1
    top_n_similar = {embedding_keys[i]: [embedding_keys[j] for j in top_n_similar_indices[i]] for i in range(len(embedding_keys))}
     
    print("Top 5 similar embeddings for each embedding:")
    for key, similar_keys in top_n_similar.items():
        print(f"{key}: {similar_keys}")

if option == options[1]:
    encoded_vals = get_encoded_vals()
    e_v = []

    for key, value in encoded_vals.items():
        e_v.append( value[0] )
    e_v = np.array( e_v )
    labels = list( encoded_vals.keys() )


    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    # Fit the model
    iso_forest.fit( e_v )
    # Predict anomalies
    anomaly_scores = iso_forest.decision_function( e_v )
    anomalies = iso_forest.predict( e_v )

    # anomalies will be -1 for anomalies and 1 for normal data points

    # Visualizing the results in 3D if using 3D embeddings
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(e_v[:, 0], e_v[:, 1], e_v[:, 2], c=anomalies, cmap='coolwarm')
    plt.colorbar(sc)
    plt.show()
