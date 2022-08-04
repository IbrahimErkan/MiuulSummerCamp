# SEABORN

# It is a data visualization library.
# It is a high-level visualization library.

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# Categorical Variable Visualization:

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()            # gender distribution

sns.countplot(x=df["sex"], data=df)
plt.show()

# matplotlib;
df["sex"].value_counts().plot(kind='bar')
plt.show()


# Numeric Variable Visualization:

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()