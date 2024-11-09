import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler

print("Cargando transacciones...")
# full file

df_cc = pd.read_csv('./credit_card_transactions-ibm_v2.csv')
df_users = pd.read_csv('./sd254_users.xls')
# df_cards = pd.read_csv('./sd254_cards.xls')
# df_usercc = pd.read_csv("./User0_credit_card_transactions.csv")
df_users["User"] = range(0, len(df_users)) # Crear columna User en df_users para index

#Limpieza de datos
df_cc['Amount'] = df_cc['Amount'].str.replace('$', '').astype(float) #dolares a float
df_cc['Hour'] = df_cc['Time'].str.split(':').str[0].astype(int)   #hora:min a hora int
df_cc.drop(columns=['Time'], inplace=True)
df_cc['Is Fraud?'] = df_cc["Is Fraud?"].map({"No": False, "Yes": True})
df_users['Per Capita Income - Zipcode'] = df_users['Per Capita Income - Zipcode'].str.replace('$', '').astype(int)
df_users['Yearly Income - Person'] = df_users['Yearly Income - Person'].str.replace('$', '').astype(int)
df_users['Total Debt'] = df_users['Total Debt'].str.replace('$', '').astype(float)
df_gigante = df_cc.merge(df_users, how="inner", on="User")   #merge de df_cc y df_users

# librera memoria plz
del df_cc
del df_users

df_model = df_gigante.head(10)


print(df_model.head())
print(df_model.columns)

chip_mapping = {
    "Swipe Transaction": 0,
    "Chip Transaction": 1,
    "Online Transaction": 2
}

gender_mapping = {
    "Male": 0,
    "Female": 1
}

df_model["Use Chip"] =  df_model['Use Chip'].replace(chip_mapping).astype(int) # chip como int
df_model["Gender"] =  df_model['Gender'].replace(gender_mapping).astype(int) # chip como int
df_model = df_model.drop(columns=['Merchant Name', 'Merchant City', 'Merchant State', 'Errors?', 'Zip', 'Person', 'Address', 'State', 'City', 'Apartment', 'Retirement Age', 'Current Age', 'Zipcode', 'Latitude', 'Longitude', 'Year', 'Month', 'Day'])
print(df_model.head())

X = df_model

print(df_model.isna().sum())

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
plt.write("./clustering/pca.png")

from sklearn.cluster import AgglomerativeClustering

# probe complete y ward con 2000 y 3000, y este da el mejor silhouette
comp_dist_3 = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=2800).fit(X)

print("Clusters Complete 3000:", comp_dist_3.n_clusters_)

from sklearn.metrics import silhouette_score
print("Silhouette de Comp 3000:\t\t", silhouette_score(X, comp_dist_3.labels_))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=comp_dist_3.labels_, alpha=0.3)
plt.write("./clustering/pca_class.png")


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
plt.write("./clustering/pca_class_mtx.png")

# caracterizacion de los clusters

w_cluster = X
print(X.head())
w_cluster["cluster"] = comp_dist_3.labels_
w_cluster["Is Fraud?"] = y


fraud_mapping = {
    "Yes": 1,
    "No": 0
}

w_cluster["Is Fraud?"] = w_cluster["Is Fraud?"].replace(fraud_mapping).astype(int)

print(w_cluster.head())

aggs = {
    "User": "count",
    "Card": "count",
    "Year": pd.Series.mode,
    "Month": pd.Series.mode,
    "Day": pd.Series.mode,
    "Hour": "mean",
    "Amount": "mean",
    "Use Chip": "count",
    "MCC": pd.Series.mode,
    "Is Fraud?": "mean",
}

clustered = w_cluster.groupby("cluster").aggregate(func=aggs)

print(clustered)

# la verdad como que no me dicen mucho los clusters

print("Listo! :)")
