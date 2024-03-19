#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Read CSV into DataFrame
df = pd.read_csv('testing/sig_genes_40pcs.csv')

# %%
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df.describe()
# %%
df.shape
# %%
# Example: Correlation heatmap
plt.figure(figsize=(10, 8), dpi=400)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
#%%
# Example: KMeans clustering
from sklearn.cluster import KMeans

# Assuming 'sigma_1' and 'sigma_e' are the features for clustering
X = df[['sigma_1', 'sigma_e']]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# Visualize clusters
sns.scatterplot(data=df, x='sigma_1', y='sigma_e', hue='cluster')
plt.title('Clusters based on sigma_1 and sigma_e')
plt.show()
# %%
