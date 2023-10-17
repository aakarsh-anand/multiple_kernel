import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True, type=str, help='bed/bim/fam prefix. Required.')
    parser.add_argument('--phen', default=459792, type=str, required=False, help='Phenotype file. Required.')
    parser.add_argument('--mStart', required=True, type=int, help='Start SNP. Required. Must be less than End SNP')
    parser.add_argument('--mEnd', required=True, type=int, help='End SNP. Required. Must be greater than Start SNP')
    parser.add_argument('--degree', required=True, type=int, help='Degree. Required.')
    parser.add_argument('--dir', required=False, default='mult_kernel_results', help='Directory for output files. Not required.')
    parser.add_argument('--filename', required=False, default='out', help='Output file name. Not required.')
    args = parser.parse_args()
    return args

def estimate_trace(X1, X2, B=50):
    M1 = X1.shape[1]
    M2 = X2.shape[1]
    n = X1.shape[0]
    vectors = np.random.multivariate_normal(np.zeros(n), np.identity(n), size=B)

    tot = 0
    for z in vectors:
        tot += np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(z.T, X1), X1.T), X2), X2.T), z)

    return (1/(B*M1*M2))*tot

if __name__ == "__main__":

    # parse arguments
    args = parseargs()

    gen = args.gen
    phen = args.phen
    mStart = args.mStart
    mEnd = args.mEnd
    D = args.degree
    dir = args.dir
    filename = args.filename

    # read data
    gendata = open_bed(f"{gen}.bed")
    phendata = pd.read_csv(phen, delim_whitespace=True)

    # create X and y
    phen_values = phendata.iloc[:,-1].values
    N = len(phen_values)
    X = gendata.read(index=np.s_[0:N, mStart:mEnd])
    
    # filter NaN
    nanfilter=~np.isnan(X).any(axis=1)
    X = X[nanfilter]
    phen_values = phen_values[nanfilter]

    # standardize genotype
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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
    outfile.write(str(np.linalg.solve(T, y)))
    outfile.close()