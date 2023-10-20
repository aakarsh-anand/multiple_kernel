import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
from scipy.linalg import pinvh

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True, type=str, help='bed/bim/fam prefix. Required.')
    parser.add_argument('--phen', required=True, type=str, help='Phenotype file. Required.')
    parser.add_argument('--covar', required=False, type=str, help='Covariate file. Not required.')
    parser.add_argument('--snp_range', nargs='+', required=True, type=int, 
                        help='SNP index range (ex. --M_range 20 35, SNPs with index 20-35 inclusive). Required.') 
    parser.add_argument('--degree', required=True, type=int, help='Degree. Required.')
    parser.add_argument('--dir', required=False, default='mult_kernel_results', help='Directory for output files. Not required.')
    parser.add_argument('--filename', required=False, default='out', help='Output file name. Not required.')
    args = parser.parse_args()
    return args

def estimate_trace(X1, X2, B=10):
    M1 = X1.shape[1]
    M2 = X2.shape[1]
    n = X1.shape[0]

    tot = 0
    for i in range(B):
        z = np.random.normal(0, 1, size=n)
        tot += np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(z.T, X1), X1.T), X2), X2.T), z)

    return (1/(B*M1*M2))*tot

def ols(X, y):
    P1 = pinvh(X.T@X)
    return P1@(X.T@y)

if __name__ == "__main__":

    # parse arguments
    args = parseargs()

    gen = args.gen
    phen = args.phen
    covar = args.covar
    snp_range = args.snp_range
    D = args.degree
    dir = args.dir
    filename = args.filename

    # read data
    gendata = open_bed(f"{gen}.bed")
    phendata = pd.read_csv(phen, delim_whitespace=True)

    c = np.array([])
    if covar != None:
        c = pd.read_csv(covar,delim_whitespace=True)
        c = c.iloc[:,2:]
        c = c.to_numpy()

    # create X and y
    phen_values = phendata.iloc[:,-1].values
    N = len(phen_values)
    X = gendata.read(index=np.s_[0:N, snp_range[0]:snp_range[1]+1])
    
    # filter NaN
    nanfilter1=~np.isnan(X).any(axis=1)
    nanfilter2=~np.isnan(c).any(axis=1)
    nanfilter=nanfilter1&nanfilter2

    X = X[nanfilter]
    phen_values = phen_values[nanfilter]

    # standardize genotype
    scaler = StandardScaler()
    X = np.unique(X, axis=1, return_index=False)
    X = scaler.fit_transform(X)
    
    # repeat steps if covar is given and regress PCs
    if c.size != 0:
        c = c[nanfilter]
        c = np.unique(c, axis=1, return_index=False)
        c = scaler.fit_transform(c)
        c = np.concatenate((np.ones((c.shape[0],1)),c),axis=1)
        phen_values -= np.matmul(c, ols(c, phen_values))

    # create kernel matrices and phenos
    kernel_matrices = [X]
    y = [np.matmul(np.matmul(np.matmul(phen_values.T, X), X.T), phen_values) / X.shape[1]]
    for d in range(2, D+1):
        poly = PolynomialFeatures(degree=(d, d), interaction_only=True, include_bias=False)
        phi = poly.fit_transform(X)
        new_pheno = np.matmul(np.matmul(np.matmul(phen_values.T, phi), phi.T), phen_values) / phi.shape[1]

        kernel_matrices.append(phi)
        y.append(new_pheno)
    y.append(np.matmul(phen_values.T, phen_values))

    print("estimating traces...")
    # create T
    T = []
    for i in range(D):
        row = []
        for j in range(D):
            row.append(estimate_trace(kernel_matrices[i], kernel_matrices[j]))
        row.append(N)
        T.append(row)
    T.append(np.full(D+1, N))

    # Exact Version

    # K = np.matmul(X, X.T) / X.shape[1]
    # poly = PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False)
    # phi = poly.fit_transform(X)
    # Q = np.matmul(phi, phi.T) / phi.shape[1]

    # T = [[np.trace(np.matmul(K, K.T)), np.trace(np.matmul(K, Q.T)), N],
    #      [np.trace(np.matmul(Q, K.T)), np.trace(np.matmul(Q, Q.T)), N],
    #      [N, N, N]]

    # yKy = np.matmul(np.matmul(phen_values.T, K), phen_values)
    # yQy = np.matmul(np.matmul(phen_values.T, Q), phen_values)
    # yy = np.matmul(phen_values.T, phen_values)

    # y = [yKy, yQy, yy]

    print("solving equation...")
    outfile = open(f"{dir}/{filename}", 'w')
    original = np.linalg.solve(T, y)
    perm_vals = {}
    for d in range(1, D+1):
        perm_vals[f"sigma_{d}"] = []
        perm_vals[f"sigma_{d}"].append(original[d-1])
    perm_vals["sigma_e"] = []
    perm_vals["sigma_e"].append(original[D])
    # for i in range(1000):
    #     y_perm = np.random.permutation(y)
    #     vals = np.linalg.solve(T, y_perm)
    #     for d in range(1, D+1):
    #         perm_vals[f"sigma_{d}"].append(vals[d-1])
    #     perm_vals["sigma_e"].append(vals[D])
    data = pd.DataFrame(perm_vals)
    data.to_csv(outfile, index=False)
    outfile.close()