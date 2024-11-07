import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

print("Cargando transacciones...")
# full file
tdf = pd.read_csv("./credit_card_transactions-ibm_v2.csv")
# tomo solo una fraccion, para probar
df_model = tdf.head(10000).copy()

print(df_model.head())

chip_mapping = {
    "Swipe Transaction": 0,
    "Chip Transaction": 1,
    "Online Transaction": 2
}

gender_mapping = {
    "Male": 0,
    "Female": 1
}

def hm_to_m(s):
    t = 0
    for u in s.split(':'):
        t = 60 * t + int(u)
    return t

df_model["Use Chip"] =  df_model['Use Chip'].replace(chip_mapping).astype(int) # chip como int
df_model["Time"] =  df_model['Time'].map(lambda x: hm_to_m(x)).astype(int) # hora a minutos a partir de 00:00
df_model["Amount"] =  df_model['Amount'].str.replace('$', '').astype(float) #  amount sin signo dolar
df_model = df_model.drop(columns=['Is Fraud?', 'Merchant Name', 'Merchant City', 'Merchant State', 'Errors?', 'Zip'])

print(df_model.head())

X = df_model

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
#plt.show()

# probe complete y ward con 2000 y 3000, y este da el mejor silhouette
comp_dist_3 = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=2300).fit(X)

print("Clusters Complete 3000:", comp_dist_3.n_clusters_)

from sklearn.metrics import silhouette_score
print("Silhouette de Comp 3000:\t\t", silhouette_score(X, comp_dist_3.labels_))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=comp_dist_3.labels_, alpha=0.3)
plt.show()

# DBSCAN con eps de 26 y 5 min samples da clusters muy similares


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def sim_matrix(features, labels):
    useful_labels = labels >= 0

    indices = np.argsort(labels[useful_labels])
    sorted_features = features[useful_labels][indices]

    d = cosine_similarity(sorted_features, sorted_features)
    return d

def plot(data, model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    fig.suptitle(f"{model.__class__.__name__}")

    ax1.scatter(data[:,0], data[:,1], c=model.labels_)

    dist = sim_matrix(data, model.labels_)
    im = ax2.imshow(dist, cmap='cividis', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax2)

plot(X_pca, comp_dist_3)
plt.show()

print("Listo! :)")
