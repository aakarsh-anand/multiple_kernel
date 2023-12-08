import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
import pandas as pd
from scipy.stats import multivariate_normal, chi2

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True, help='bed/bim/fam prefix. Required')
    parser.add_argument('--phen', required=True, help='phenotype file. Required.')
    parser.add_argument('--snp_range', nargs='+', default=[0, 459791], type=int, required=False, 
                        help='SNP index range (ex. --snp_range 20 35, SNPs with index 20-35 inclusive). Not required.')
    parser.add_argument('--null', required=True, help='Null csv')
    parser.add_argument('--alt', required=True, help='Alt csv')
    parser.add_argument('--dir', required=False, default='mult_kernel_results', help='Directory for output files. Not required.')
    parser.add_argument('--filename', required=False, default='out', help='Output file name. Not required.')
    args = parser.parse_args()
    return args

def loglik(X, y, M, degree, sigmas):
    N = X.shape[0]
    mu = np.zeros(N)

    # construct kernels with corresponding components
    K = sigmas[0] * (np.matmul(X, X.T) / (M[1]-M[0]+1))
    I = sigmas[-1] * (np.identity(N))

    Q = np.zeros((N, N))
    if degree > 1:
        for k in range(degree-1):
            poly = PolynomialFeatures((k+2, k+2), interaction_only=True, include_bias=False)
            phi = poly.fit_transform(X)
            scaler = StandardScaler()
            phi = scaler.fit_transform(phi)
            nonlinear = np.matmul(phi, phi.T) / phi.shape[1]
            Q += (sigmas[k+1] * nonlinear)

    # log likelihood
    mat = K+Q+I
    return multivariate_normal.logpdf(y, mu, mat)

if __name__ == "__main__":

    # parse arguments
    args = parseargs()

    gen = args.gen
    phen = args.phen
    M = args.snp_range
    null = args.null
    alt = args.alt
    dir = args.dir
    filename = args.filename

    d1 = pd.read_csv(null)
    d3 = pd.read_csv(alt)

    # load genotype matrix
    bedfile = f"{gen}/filter4.bed"
    phen = pd.read_csv(phen,delim_whitespace=True).iloc[:,2].values

    #i = np.arange(0, N)
    snps = np.arange(M[0], M[1]+1)
    bed = open_bed(bedfile)
    X = bed.read(index=np.s_[:,snps])

    # impute with average value
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    # standardize genotype matrix
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    s1 = d1.iloc[0,:].values
    s3 = d3.iloc[0,:].values
    s1[s1<0]=0
    s3[s3<0]=0
    l0 = loglik(X,phen,M,degree=1,sigmas=s1)
    l1 = loglik(X,phen,M,degree=3,sigmas=s3)

    lrt = -2 * (l0-l1)
    print("pval: " + str(1-chi2.cdf(lrt, 1)))

    outfile = open(f"{dir}/{filename}", 'w')
    outfile.write(str(1-chi2.cdf(lrt, 1)))
    outfile.close()
# %%
