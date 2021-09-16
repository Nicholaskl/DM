# %%
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_name = 'kddcup.data_10_percent_corrected_10000_samples.arff'
data,metadata = arff.loadarff(file_name)

# create a data frame - we're only looking at the 'count' attribute
df = pd.DataFrame(data)['count']

# Mean
mean = df.mean()

# Standard Deviation
std = df.std()

# Sample Sizing
sample_sizes = [10,20,50,100,200,500,1000,2000,5000,10000]

repeat = 1000 # Times repeated -> says 10 in task
z_scores = []

# Calculate for every sample size
for n in sample_sizes:
    zn = 0

    # Repeat a specific amount of times for an average
    for i in range(repeat):
        # First sample the dataset
        df_sample = df.sample(n, axis=0)

        # calculate the mean for the sample and the s.d. 
        en = df_sample.mean()
        zn += abs(en - mean) / std

    # find the avg z score for the sample
    z_avg = zn / repeat
    z_scores.append(z_avg)
        
# Plot sample size vs the z score
plt.plot(sample_sizes, z_scores)
plt.ylabel('ratio')
plt.xlabel('sample size')
plt.show()

# %%
