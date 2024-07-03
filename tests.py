from tensorflow.keras.models import load_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from gencovvec import get_labelled_coverage_vectors
from genemb import get_encoded_vals 
from sklearn.ensemble import IsolationForest

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

# e_v = []
# for key, value in encoded_vals.items():
#     e_v.append( value[0] )
# e_v = np.array( e_v )
# labels = list( encoded_vals.keys() )
# 
# 
# 
# iso_forest = IsolationForest(contamination=0.1)
# iso_forest.fit( e_v )
# anomalies = iso_forest.predict( e_v )
# print(anomalies)




# tsne = TSNE(n_components = 3)
# embeddings_3d = tsne.fit_transform( e_v )
# df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
# df['label'] = labels
# 
# # Plot with Plotly
# fig = px.scatter_3d(df, x='x', y='y', z='z', text='label')
# fig.update_traces(marker=dict(size=5), selector=dict(mode='markers+text'))
