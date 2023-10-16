import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
import pandas as pd
import sys

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', required=True, type=int, help='Sample size. Required.')
    parser.add_argument('--M', default=459792, type=int, required=False, help='SNPs. Not required.')
    parser.add_argument('--degree', required=True, type=int, help='Degree. Required.')
    parser.add_argument('--sigmas', required=True, nargs='+', type=float, help='Variance components. Required.')
    parser.add_argument('--dir', required=False, default='sim_phenos', help='Directory for output files. Not required.')
    parser.add_argument('--filename', required=False, default='sim', help='Output file name. Not required.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse arguments
    args = parseargs()

    N = args.N
    M = args.M
    degree = args.degree
    sigmas = args.sigmas
    directory = args.dir
    filename = args.filename

    if len(sigmas) != degree + 1:
        sys.exit("Number of components does not match degree.")

    # load genotype matrix
    bedfile = "/u/project/sgss/UKBB/data/cal/filter4.bed"
    ids1 = pd.read_csv("/u/project/sgss/UKBB/data/cal/filter4.fam",delim_whitespace=True,header=None).iloc[:,0].values
    ids2 = pd.read_csv("/u/project/sgss/UKBB/data/cal/filter4.fam",delim_whitespace=True,header=None).iloc[:,1].values

    i = np.arange(0, N)
    snps = np.arange(0, M)
    bed = open_bed(bedfile)
    X = bed.read(index=np.s_[i,snps])

    # impute with average value
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    # standardize genotype matrix
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    mu = np.zeros(N)

    # construct kernels with corresponding components
    K = sigmas[0] * (np.matmul(X, X.T) / M)
    I = sigmas[-1] * (np.identity(N))

    Q = np.zeros((N, N))
    if degree > 1:
        for k in range(degree-1):
            poly = PolynomialFeatures((k+2, k+2), interaction_only=True, include_bias=False)
            phi = poly.fit_transform(X)
            nonlinear = np.matmul(phi, phi.T) / phi.shape[1]
            Q += sigmas[k+1] * nonlinear

    # draw phenotype from multivariate normal distribution
    y = np.random.multivariate_normal(mu, K + Q + I)
    ids1 = ids1[0:N]
    ids2 = ids2[0:N]

    # save phenotype to file
    df = pd.DataFrame(list(zip(ids1, ids2, y)))
    df.to_csv(directory + '/' + filename, sep=' ', header=["FID", "IID", "pheno"], index=False, mode='w')
