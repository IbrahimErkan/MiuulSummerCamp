# PANDAS

# Pandas Series
# Reading Data (Veri Okuma)
# Quick Look at Data (Veriye Hızlı Bakış)
# Selection in Pandas (Pandas'ta Seçim İşlemleri)
# Aggregation & Grouping (Toplulaştırma ve Gruplama)
# Apply & Lambda
# Birleştirme (Join) İşlemleri


# Pandas Series

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index   # .index() the index of the pandas series is created with
s.dtype   # .dtype() returns the type of pandas series with
s.size    # .size() returns the size of the pandas series with
s.ndim    # .ndim() returns the dimensions of the pandas series with
s.values  # .values() returns the values of the pandas series with
type(s.values)
s.head(3)          # .head(3) returns 3 at the beginning of the pandas series with
s.tail(3)          # .tail() returns 3 at the end of the pandas series with


# Reading Data (Veri Okuma)

import pandas as pd
df = pd.read_csv(".......")     #ctrl + f
df.head


# Quick Look at Data (Veriye Hızlı Bakış)

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape      # .shape() returns the dimensions of the dataset
df.info()     # .info() returns the information of the dataset
df.columns    # .columns() returns the columns of the dataset
df.index
df.describe()    # .describe() returns the summary information of the dataset with. Transposed with T
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()


# Selection in Pandas (Pndas'ta Seçim İşlemleri)

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0).head()

delete_index = [1, 3, 5, 7]
df.drop(delete_index, axis=0).head(10)

# df = df.drop(delete_index, axis=0)
# df.drop(delete_index, axis=0, inplace=True)

# Convert Variable to Index (Değişkeni Indexe Çevirme)

df["age"].head()
df.age.head()

df.index = df["age"]
print(df.index)

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

# Convert Index to Variable (İndexi Değişkene Çevirme)

df.index
#method 1
df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)

#method 2
df.reset_index().head()
df = df.reset_index()
df.head()

# Operations on Variables(Columns)

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())     # pandas Series

df[["age"]].head()
type(df[["age"]].head())   # pandas DataFrame

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2
print(df.head())
df["age3"] = df["age"] / df["age2"]
df.head()

df.drop("age", axis=1).head()

col_names = ["age", "adult_male", "alive"]
df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head()   # age olanları seç
df.loc[:, ~df.columns.str.contains("age")].head()  # age dışındakileri seç demek


# iloc & loc

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3]
df.iloc[0, 0]

# df.iloc[0:3, "age"]   # "age" yerine integer bir değişken gelmeli

# loc: label based selection
df.loc[0:3]

df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


# Conditional Selection (Koşullu Seçim)

df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, "class"].head()
df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & (df["embark_town"] == "Cherbourg"),
       ["age", "class", "embark_town"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()

df_new["embark_town"].value_counts()


# Aggregation & Grouping (Toplulaştırma ve Guruplama)

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                                          "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                                                 "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})


# Pivot Table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])       # .cut : convert numeric variable to categorical variable
df.head()

df.pivot_table("survived", "sex", "new_age")
df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)       #okunurluluğu artırıyor


# Apply & Lambda

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
df.head()

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

# apply & lambda
df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean() / x.std())).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.head()


# Birleştirme (Join) İşlemleri

# concat:

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)               # ignore_index=True

# merge:

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")   # çalışanlara göre sıralama

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})
print(df4)

pd.merge(df3, df4)

# concat & merge : combination operations are performed with the function.
