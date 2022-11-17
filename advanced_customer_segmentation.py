import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from kmodes.kprototypes import KPrototypes 

df = pd.read_csv('segmentation-data.csv')

df.head()

# Imprime o somatório de dados faltantes de cada coluna
df.isnull().sum()

# Normalização 

df_temp = df[['ID','Age','Income']]

print(df_temp) 

# Scaller
scaler = MinMaxScaler()

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

# ID nao é usado para a análise
df = df.drop(['ID'], axis=1)

# Converter Age e Income para float
mark_array =df.values
mark_array[:, 2] = mark_array[:, 2].astype(float)
mark_array[:, 4] = mark_array[:, 4].astype(float)

kproto = KPrototypes(n_clusters=5, verbose=3, max_iter=50)

# as colunas consideradas categorical não entram no calculo dos centroids 
clusters = kproto.fit_predict(mark_array, categorical = [0, 1, 3, 5, 6])

print(f'os centros são: {kproto.cluster_centroids_}')
print(f'o comprimento dos centroids é: {len(kproto.cluster_centroids_)}')

cluster_dict = []

for c in clusters:
    cluster_dict.append(c)

df['cluster'] = cluster_dict

df[['ID','Age','Income']] = df_temp 

df[df['cluster'] == 0].head(10)

colors = ['green','red', 'gray', 'orange', 'cyan']

fig, ax = plt.subplots(figsize=(12,12))
ax.set_xlabel('Age')
ax.set_ylabel('Income')

for i, col in zip(range(5), colors):
    dftemp = df[df.cluster==i]
    ax.scatter(dftemp.Age, dftemp['Income'], color=col, alpha=0.5)



plt.legend(['cluster1','cluster2','cluster3','cluster4','cluster5'])
plt.savefig('clusters.png',dpi=300)
plt.show()
