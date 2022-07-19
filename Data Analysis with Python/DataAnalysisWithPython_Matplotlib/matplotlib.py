# DATA VISUALIZATION : MATPLOTLIB & SEABORN

# MATPLOTLIB:

'''
* It is the ancestor of data visualization in Python.
* It is a low-level data visualization tool.

* Categorical Variable: column(sütun) chart(grafik), countplot, bar plot
* Numeric Variable: histogram, boxplot
'''

# Categorical Variable Visualization:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()
df["sex"].value_counts().plot(kind='bar')
plt.show()


# Numeric Variable Visualization:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


# Matplotlib Features:   (Matplotlib'in Özellikleri)

# 1-) plot

# Veriyi görselleştirmek için kullandığımız fonksiyonlardan birisi

x = np.array([1, 8])
y = np.array([0, 150])
plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()


# 2-) marker:

y = np.array([13, 28, 11, 100])
plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='*')
plt.show()

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']


# 3-) line:

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")      # linestyle= "dotted", "dashed"
plt.show()

# multiple lines:

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()


# 4-) labels:

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

# title:
plt.title("This is main title")

# axis naming
plt.xlabel("X-axis naming")

plt.ylabel("Y-axis naming")

plt.grid()
plt.show()


# 5-) subplots:  (çoklu grafik)

# plot 1:
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)       # satır, sütun, kaçıncı grafik old.
plt.title("1")
plt.plot(x, y)

# plot 2:
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3:
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()