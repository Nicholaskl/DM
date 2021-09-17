#%%
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

df=pd.read_csv('D:\CURTIN\YEAR3\Sem2\DM\Assignment\data2021.student.csv', sep=',')

profile = ProfileReport(df, title="Pandas Profiling Report")

profile.to_file("your_report.html")









# %%
