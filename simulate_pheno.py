#%%
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
import pandas as pd
import sys
#%%
bedfile = "/u/project/sgss/UKBB/data/cal/filter4.bed"
ids1 = pd.read_csv("/u/project/sgss/UKBB/data/cal/filter4.fam",delim_whitespace=True,header=None).iloc[:,0].values
ids2 = pd.read_csv("/u/project/sgss/UKBB/data/cal/filter4.fam",delim_whitespace=True,header=None).iloc[:,1].values
#%%
N = 5000
M = 10

i = np.arange(0, N)
snps = np.arange(0, M)
#bed = open_bed(bedfile)
#%%
#file1 = open(f"/u/home/a/aanand2/multiple_kernel/phenos/temp.pheno", 'a')
#X = bed.read(index=np.s_[i,snps])
X = np.random.randint(3, size=(N, M))

# impute with average value
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

idsblock1 = ids1[0:N]
idsblock2 = ids2[0:N]
#%%
scaler = StandardScaler()
X = scaler.fit_transform(X)
mask = (X != 0).any(axis=0)
X = X[:, mask]
#print(X)
#%%
sigma_g = 0.2
sigma_q = 0.01
sigma_e = 0.5

mu = np.zeros(N)
K = np.matmul(X, X.T) / M
Q = polynomial_kernel(X, degree=2) / M
#poly = PolynomialFeatures((2, 2), interaction_only=True, include_bias=False)
#Q = poly.fit_transform(X) / M
I = np.identity(N)

#print(K)
#%%
y = np.random.multivariate_normal(mu, sigma_g * K + sigma_q * Q + sigma_e * I)
df = pd.DataFrame(list(zip(idsblock1, idsblock2, y)))
#df.to_csv(file1, sep=' ', header=["FID", "IID", "pheno"], index=False, mode='w')
#file1.close()
# %%
K2 = np.matmul(K, K.T)
KQ = np.matmul(K, Q.T)
QK = np.matmul(Q, K.T)
Q2 = np.matmul(Q, Q.T)

T = [[np.trace(K2), np.trace(KQ), np.trace(K)],
    [np.trace(QK), np.trace(Q2), np.trace(Q)],
    [np.trace(K), np.trace(Q), np.trace(I)]]

yKy = np.matmul(np.matmul(y.T, K), y)
yQy = np.matmul(np.matmul(y.T, Q), y)
yy = np.matmul(y.T, y)

y = [yKy, yQy, yy]
#%%
sigma = np.linalg.solve(T, y)
print(sigma)
# %%
