# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    'uniform.csv', 
    sep=',',
    header=None, # There are no column names... 
)

vals = np.array(df.loc[:9,:].values, dtype=np.int64)
cols = len(vals[0])

# use scipy's pdist as reference to compare your implementation
from scipy.spatial import distance
min = distance.squareform(distance.pdist(vals, 'minkowski', p=1)).astype(int)
euc = distance.squareform(distance.pdist(vals, 'euclidean')).astype(int)
che = distance.squareform(distance.pdist(vals,'chebyshev' )).astype(int)

def lp_distance(x1,x2,p=2):
    """
        lp distance between two numpy vectors x1, x2
        p is either 1, 2, or np.inf
    """
    return np.linalg.norm(x1 - x2, p)

def lp_distance_matrix(X,p):
    """
        lp distance matrix of all rows in X
    """    
    export = []
    for x in X:
        row = []
        for x1 in X:
            # Check each row against every other row
            row.append(lp_distance(x, x1, p))
        
        export.append(row)
    return np.array(export, dtype=np.int64)

min_two = lp_distance_matrix(vals, 1)
euc_two = lp_distance_matrix(vals, 2)
che_two = lp_distance_matrix(vals, np.inf)


# %%

# Contrast C = (Dmax-Dmin) / mu

def contrast(X, p=1, r=None):
    """
        compute the contrast of a set of points in X         
        given order p and dimensionality r       
    """
    contrasts = []
    empty = np.zeros(r)

    for i in range (0, 500):
        selected_columns = np.random.choice(range(cols), r, replace=False)
        x = X[:, selected_columns]    

        distances = []
        for row in x:
            distances.append(lp_distance(row, empty, p))
            
        
        # This will give us an array of distances
        # distances = lp_distance_matrix(x, p)

        # print(distances)

        contrasts.append((np.max(distances) - np.min(distances)) / np.average(distances))
    return np.average(contrasts)

# %%

## Sonny code to check against the theoretical values

for p in [1,2,np.inf]:
    contrast_scores = np.zeros((cols,1),dtype=float)
    for r in range(cols):
        contrast_scores[r]=contrast(vals,p,r+1)
    plt.plot(range(cols),contrast_scores)
    plt.ylabel('contrast')
    plt.xlabel('dimensions')
    plt.title('p='+str(p))
    plt.show() 

contrast_scores = np.zeros((cols,1),dtype=float)
for r in range(cols):
    contrast_scores[r]=1/np.sqrt(r+1)
plt.plot(range(cols),contrast_scores)
plt.ylabel('contrast')
plt.xlabel('dimensions')
plt.title('theory')
plt.show() 

# %%
