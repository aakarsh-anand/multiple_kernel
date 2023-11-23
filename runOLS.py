import argparse
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from bed_reader import open_bed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True, type=str, help='bed/bim/fam prefix. Required.')
    parser.add_argument('--snplist', required=True, type=str, help='SNPs to test. Required.')
    parser.add_argument('--phen', required=True, type=str, help='Phenotype file. Required.')
    parser.add_argument('--covar', required=False, type=str, help='Covariate file. Not required.')
    parser.add_argument('--dir', required=True, type=str, help='output dir. Required.')
    parser.add_argument('--filename', required=True, type=str, help='file name. Required.')
    args = parser.parse_args()
    return args

def ols(X, y):
    P1 = pinvh(X.T@X)
    return P1@(X.T@y)

def match_indices(bimPath,fPath):
    bim = pd.read_csv(bimPath,delim_whitespace=True,header=None)
    f_df = pd.read_csv(fPath,header=None)
    Indices = bim.loc[bim.iloc[:,1].isin(f_df.values.flatten())].index.values
    return Indices

if __name__ == "__main__":
    
    args = parseargs()

    gen = args.gen
    snplist = args.snplist
    phen = args.phen
    covar = args.covar
    dir = args.dir
    filename = args.filename

    if os.stat(snplist).st_size == 0:
        with open(f"{dir}/{filename}", "w") as f:
            f.write(f"Trait: {phen}\n")
            f.write(f"MSE: 0\n")
            f.write(f"Pearson: 0")
        exit()

    # read data
    gendata = open_bed(f"{gen}.bed")
    phendata = pd.read_csv(phen, delim_whitespace=True)
    phen_values = phendata.iloc[:,-1].values
    yfilter = (phen_values != -9) & (~np.isnan(phen_values))
    phen_values = phen_values[yfilter]

    # read covar
    c = np.array([])
    if covar != None:
        c = pd.read_csv(covar,delim_whitespace=True)
        c = c.iloc[:,2:]
        c = c.to_numpy()
        c = c[yfilter]

    N = len(phen_values)
    fIndices = match_indices(f"{gen}.bim", snplist)
    X = gendata.read(index=np.s_[yfilter,fIndices])
    
    # filter NaN
    nanfilter1=~np.isnan(X).any(axis=1)
    if covar != None:
        nanfilter2=~np.isnan(c).any(axis=1)
        nanfilter=nanfilter1&nanfilter2
    else:
        nanfilter=nanfilter1

    X = X[nanfilter]
    phen_values = phen_values[nanfilter]

    # standardize genotype
    scaler = StandardScaler()
    X = np.unique(X, axis=1, return_index=False)
    X = scaler.fit_transform(X)

    if c.size != 0:
        c = c[nanfilter]
        c = np.unique(c, axis=1, return_index=False)
        c = scaler.fit_transform(c)
        c = np.concatenate((np.ones((c.shape[0],1)),c),axis=1)
        o = ols(c, phen_values)
        phen_values -= np.matmul(c, o)

    OLS = ols(X, phen_values)
    y_pred = np.matmul(X, OLS)

    mse = mean_squared_error(phen_values, y_pred)
    pearson = pearsonr(y_pred, phen_values)
    print(f"MSE: {mse}")
    print(f"Pearson: {pearson.statistic}")

    with open(f"{dir}/{filename}", "w") as f:
        f.write(f"Trait: {phen}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"Pearson: {pearson.statistic}")