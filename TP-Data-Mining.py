#TRABAJO PRACTICO PARA MINERIA DE DATOS
#INTEGRANTES:
#MATEO ROVERE
#VALENTIN DALMAU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

df = pd.read_csv('Crop_recommendation.csv')
X = df.drop(columns=['label'])
y = df['label']

#2)

df.isna().sum()

df.describe()

df.info()

y.value_counts()

plt.figure(figsize=(10, 6)) 
sns.boxplot(data=df)  
plt.title('Gráfico de Caja por Variable') 
plt.show()
# En este grafico vimos que habia dos clases enteras que aparecian como valores atipicos en la columna de Sodio, 
# pero al ser clases enteras no los sacamos

sns.heatmap(X.corr(), annot=True)
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.boxplot(data=X_scaled)
plt.title('Gráfico de Caja por Variable')
plt.show()

df[df['K']>150]['label'].unique()

df[df['rainfall']>200]['label'].unique()

df[df['P']>125]['label'].unique()

X_scaled['label'] = y

X_scaled[X_scaled['ph']<-2]['label'].unique()

X_scaled[X_scaled['ph']>2]['label'].unique()

df[df['label'] == 'mothbeans']['ph']

X_scaled[X_scaled['temperature']>2]['label'].unique()

X_scaled[X_scaled['temperature']<-2]['label'].unique()

#Se puede observar como la mayor parte de los "valores atípicos" tienen sentido, 
# porque cada clase tiene una media y un Desvío estandar distinto para cada atributo. 
#El único caso realmente atípico es el de los mothbeans, con PHs muy bajos y muy altos.
#Aunque decidimos no modificar, ni extraerlos por falta de conocimiento tecnico en el tema.

#3) PCA

pca = PCA(n_components=X.shape[1])
pca_features = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(
    data=pca_features,
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
pca_df['Label'] = y

pca_df

def acumular(numbers):
     sum = 0
     var_c = []
     for num in numbers:
        sum += num
        var_c.append(sum)
     return var_c
var_c = acumular(pca.explained_variance_ratio_)
pca_rtd = pd.DataFrame({'Eigenvalues':pca.explained_variance_, 'Proporción de variancia explicada':pca.explained_variance_ratio_, 'Proporción acumulado de variancia explicada': var_c})
pca_rtd

eigenvectors = pca.components_
print("Eigenvectors (Principal Components):")
print(eigenvectors)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='steelblue')
plt.title('Scree Plot')
plt.xlabel('Componentes principales')
plt.ylabel('Proporción de variancia explicada')
plt.show()

#Podriamos elegir 4 PC, para usar la regla de la Proporción de variancia acumulada

plt.bar(range(1,8), pca.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(1,8), np.cumsum(pca.explained_variance_ratio_),
         where='mid',
         color='red')
plt.ylabel('Proporción de variancia explicada')
plt.xlabel('Componente principales')
plt.show()

corr = pca_df[['PC1', 'PC2', 'PC3', 'PC4']].corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws = {'size': 6}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()


features = df.drop(columns=['label']).columns.to_list()

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(pca_features, x=0, y=1, color = pca_df["Label"],  labels={'color': 'Label'} )
fig.update_layout(title = "Biplot",width = 1200,height = 600)
fig.show()
fig = px.scatter_3d(pca_features, x=0, y=1, z=2,
              color=pca_df["Label"],  labels={'color': 'Label'})
fig.show()

#4) Isomap


#vimos con diferentes valores en "n_neighbors" y con 5 nos parecio la mejor opcion
isomap_df = Isomap(n_neighbors=5, n_components=2)
isomap_df.fit(X_scaled)

projections_isomap = isomap_df.transform(X_scaled)

#fig = px.scatter_3d(
#    projections_isomap, x=0, y=1, z=2,
#    color=df['label'], labels={'color': 'label'}
#)
fig = px.scatter(projections_isomap, x=0, y=1,color=df['label'], labels={'color': 'label'})
fig.show()

#5) t-SNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# Graficar los resultados de t-SNE en 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1')
plt.xlabel('Componente 1 (t-SNE)')
plt.ylabel('Componente 2 (t-SNE)')
plt.title('Resultados de t-SNE en 2D')
plt.legend(title='Clases')
plt.grid(True)
plt.show()

#6) K-MEANS

inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.xticks(np.arange(1, 11))
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_scaled) #Entrenamos el modelo

# El metodo labels_ nos da a que cluster corresponde cada observacion
df['Cluster KMeans'] = kmeans.labels_
df.head()

pca = PCA(n_components=3)
components_principales = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    components_principales[:, 0],
    components_principales[:, 1],
    components_principales[:, 2],
    c=df['Cluster KMeans'],
    cmap='rainbow',
    alpha=0.5
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Visualización de Clústeres utilizando PCA con 3 componentes principales')

plt.show()


#Con el GAP nos da un valor de K bastante mas elevado que por el metodo del codo,
# pero con el metodo del codo notamos una mejor clusterizacion, capaz por que es menos estricto

gs_obj = OptimalK(n_jobs=1, n_iter= 10)
n_clusters = gs_obj(X, n_refs=50, cluster_array=np.arange(1, 15))
print('Optimal number of clusters: ', n_clusters)

# aunque aca podemos ver que nos dice que es 14 el optimal k,
#aunque si aumentamos el valor maximo de cluster_array,
# va a aumentar tambien el optimalk


#7) CLUSTERING JERARQUICO

Z = linkage(X_scaled, "ward")

dendrogram(Z)
plt.show()

distancias=[]
for i in range(1, 30):
    clustering = AgglomerativeClustering(n_clusters=i)
    clustering.fit(X_scaled)

    # Calculo la matriz de distancias entre puntos
    pairwise_distances = cdist(X_scaled, X_scaled, 'euclidean')

    # Calculo la distancia total entre los clusters
    distancia_total = 0
    for j in range(i):
        cluster_indices = np.where(clustering.labels_ == j)
        distancia_total += pairwise_distances[cluster_indices][:, cluster_indices].sum()

    distancias.append(distancia_total)


plt.plot(range(1, 30), distancias, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Distancia Total')
plt.title('Método del Codo para Clustering Jerárquico')
plt.show()


#podemos encontrar un 'codo' en 5

def calculate_silhouette(X_scaled, k):
    clustering = AgglomerativeClustering(n_clusters=k)
    cluster_assignments = clustering.fit_predict(X_scaled)
    df['Cluster'] = cluster_assignments
    silhouette_avg = silhouette_score(X_scaled, cluster_assignments)
    return silhouette_avg
max_k = 30

silhouette_scores = []
for k in range(2, max_k + 1):
    silhouette_avg = calculate_silhouette(X_scaled, k)
    silhouette_scores.append(silhouette_avg)
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silhouette')
plt.title('Coeficiente de Silhouette para determinar el número óptimo de clusters')
plt.show()

#utilizando silhouette, podemos ver que el numero optimo de clusters es 2

gs_obj = OptimalK(n_jobs=1, n_iter=20)
n_clusters = gs_obj(X_scaled.astype('float'), n_refs=60,
cluster_array=np.arange(2, 10))
print('Optimal number of clusters: ', n_clusters)

#aunque en el gap me dice que es 9 el optimalk