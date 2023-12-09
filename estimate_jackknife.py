import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
from scipy.linalg import pinvh
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True, type=str, help='bed/bim/fam prefix. Required.')
    parser.add_argument('--phen', required=True, type=str, help='Phenotype file. Required.')
    parser.add_argument('--covar', required=False, type=str, help='Covariate file. Not required.')
    parser.add_argument('--snp_range', nargs='+', required=True, type=int, 
                        help='SNP index range (ex. --M_range 20 35, SNPs with index 20-35 inclusive). Required.') 
    parser.add_argument('--degree', required=True, type=int, help='Degree. Required.')
    parser.add_argument('--B', required=False, default=50, type=int, help='Number of random vectors for trace approximation. Not Required.')
    parser.add_argument('--J', required=False, default=100, type=int, help='Number of jack-knifes. Not Required.')
    parser.add_argument('--dir', required=False, default='mult_kernel_results', help='Directory for output files. Not required.')
    parser.add_argument('--filename', required=False, default='out', help='Output file name. Not required.')
    args = parser.parse_args()
    return args

def estimate_trace(X1, X2, B):
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

def jackknife_indices(num_samples, num_knives): 
    if (num_knives > num_samples):
        raise Exception("Too many knives")
    indices = []
    samples = list(range(num_samples))
    for split in np.array_split(samples, num_knives, axis=0):
        knife = [m for m in samples if m not in split]
        indices.append(knife)
    indices.append(samples)
    return indices

def process_chunk(indices, X, phen_values, c):
    result = {}
    X_m = X[indices, :]
    phen_values_m = phen_values[indices]
    
    # standardize genotype
    scaler = StandardScaler()
    X_m = np.unique(X_m, axis=1, return_index=False)
    X_m = scaler.fit_transform(X_m)
    
    # repeat steps if covar is given and regress PCs
    if c.size != 0:
        c_m = c[indices]
        c_m = np.unique(c_m, axis=1, return_index=False)
        c_m = scaler.fit_transform(c_m)
        c_m = np.concatenate((np.ones((c_m.shape[0],1)),c_m),axis=1)
        o = ols(c_m, phen_values_m)
        phen_values_m -= np.matmul(c_m, o)

    # create kernel matrices and phenos
    kernel_matrices = [X_m]
    y = [np.matmul(np.matmul(np.matmul(phen_values_m.T, X_m), X_m.T), phen_values_m) / X_m.shape[1]]
    for d in range(2, D+1):
        poly = PolynomialFeatures(degree=(d, d), interaction_only=True, include_bias=False)
        phi = poly.fit_transform(X_m)
        phi = scaler.fit_transform(phi)
        new_pheno = np.matmul(np.matmul(np.matmul(phen_values_m.T, phi), phi.T), phen_values_m) / phi.shape[1]

        kernel_matrices.append(phi)
        y.append(new_pheno)
    y.append(np.matmul(phen_values_m.T, phen_values_m))

    # create T
    T = []
    for i in range(D):
        row = []
        for j in range(D):
            row.append(estimate_trace(kernel_matrices[i], kernel_matrices[j], B))
        row.append(N)
        T.append(row)
    T.append(np.full(D+1, N))

    sol = np.linalg.solve(T, y)
    for d in range(1, D+1):
        result[f"sigma_{d}"] = sol[d-1]
    result["sigma_e"] = sol[D]

    return result

if __name__ == "__main__":

    np.random.seed(0)

    # parse arguments
    args = parseargs()

    gen = args.gen
    phen = args.phen
    covar = args.covar
    snp_range = args.snp_range
    D = args.degree
    B = args.B
    J = args.J
    dir = args.dir
    filename = args.filename

    # read data
    gendata = open_bed(f"{gen}.bed")
    phendata = pd.read_csv(phen, delim_whitespace=True)
    fid = phendata.iloc[:, 0].values
    iid = phendata.iloc[:, 1].values
    phen_values = phendata.iloc[:,-1].values

    yfilter = np.array([])
    fam = pd.read_csv(f"{gen}.fam", delim_whitespace=True, header=None)
    fid_all = fam.iloc[:,0].values
    iid_all = fam.iloc[:,1].values
    for i in range(len(fid)):
        idx1 = np.where(fid_all == fid[i])[0]
        idx2 = np.where(iid_all == iid[i])[0]
        if idx1 == idx2:
            yfilter = np.append(yfilter, i)
            
    idx = (phen_values != -9) & (~np.isnan(phen_values))
    yfilter = yfilter[idx]
    yfilter = [int(x) for x in yfilter]

    phen_values = phen_values[yfilter]

    c = np.array([])
    if covar != None:
        c = pd.read_csv(covar,delim_whitespace=True)
        c = c.iloc[:,2:]
        c = c.to_numpy()
        c = c[yfilter]

    # create X and y

    X = gendata.read(index=np.s_[yfilter, snp_range[0]:snp_range[1]+1])
    
    # filter NaN
    nanfilter1=~np.isnan(X).any(axis=1)
    if covar != None:
        nanfilter2=~np.isnan(c).any(axis=1)
        nanfilter=nanfilter1&nanfilter2
    else:
        nanfilter=nanfilter1

    X = X[nanfilter]
    phen_values = phen_values[nanfilter]
    if c.size != 0:
        c = c[nanfilter]

    N = len(phen_values)

    # Create a partial function with fixed values for additional variables
    partial_process_chunk = partial(process_chunk, X=X, phen_values=phen_values, c=c)

    knife_indices = jackknife_indices(N, J)
    # Process each chunk in parallel using the partial function
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(partial_process_chunk, knife_indices))

    df = pd.DataFrame(results)

    # Compute mean and std for each column
    column_stats = pd.DataFrame({
        'mean': df.iloc[:J, :].mean(),
        'var': df.iloc[:J, :].std() * ((J-1)/np.sqrt(J))
    })

    vals = pd.concat([df.iloc[J, :], column_stats], axis=1)
    
    vals.columns = ["obs", "mean", "std"]
    vals['z'] = vals.apply(lambda row: row["mean"]/row["std"], axis=1)
    vals['pvalue'] = vals.apply(lambda row: 2*(1 - stats.norm.cdf(row["z"], 1)), axis=1)

    outfile = open(f"{dir}/{filename}", 'w')
    vals.to_csv(outfile, index=True)
    outfile.close()