import argparse
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from bed_reader import open_bed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

    if len(c) != 0:
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, phen_values, c, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, phen_values, test_size=0.3, random_state=42)

    # standardize genotype
    scaler = StandardScaler()
    #X_train = np.unique(X_train, axis=1, return_index=False)
    #X_test = np.unique(X_test, axis=1, return_index=False)
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    if c.size != 0:
        c = c[nanfilter]
        c = np.unique(c, axis=1, return_index=False)
        c = scaler.fit_transform(c)
        c = np.concatenate((np.ones((c.shape[0],1)),c),axis=1)

        o_train = ols(c_train, y_train)
        y_train -= np.matmul(c, o_train)

        o_test = ols(c_test, y_test)
        y_test -= np.matmul(c, o_test)

    X_train_scale = np.concatenate((np.ones((X_train_scale.shape[0],1)),X_train_scale),axis=1)
    X_test_scale = np.concatenate((np.ones((X_test_scale.shape[0],1)),X_test_scale),axis=1)

    OLS_train = ols(X_train_scale, y_train)
    y_pred = np.matmul(X_test_scale, OLS_train)

    mse = mean_squared_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"Pearson: {pearson.statistic}")

    with open(f"{dir}/{filename}", "w") as f:
        f.write(f"Trait: {phen}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"Pearson: {pearson.statistic}")