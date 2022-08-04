# ADVANCED FUNCTIONAL EXPLORATORY DATA ANALYSIS  (EDA)

# 1. General Picture
# 2. Analysis of Categorical Variables
# 3. Analysis of Numerical Variables
# 4. Analysis of Target Variable
# 5. Analysis of Correlation


# 1. General Picture

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe()
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ###################")
    print(dataframe.head(head))
    print("################## Tail ###################")
    print(dataframe.tail(head))
    print("################## NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles ##############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# new dataset
df = sns.load_dataset("tips")
check_df(df)

df = sns.load_dataset("flights")
check_df(df)



# 2. Analysis of Categorical Variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/advertising.csv")
df = sns.load_dataset("titanic")
print(df.head())

print(df["embarked"].value_counts())
print(df["sex"].unique())  # unique() fonksiyonu ile birbirinden farklı kaç kategori olduğunu gösterir.
print(df["class"].nunique())  # kaç sınıf olduğunu verir.

catCols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
print(catCols)  # Tip olarak object, category veya bool olan sütunların listesi

numButCat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
# Tip olarak int veya float olan sütunların listesi ve 10'dan küçük nunique değerleri olan sütunların listesis
print(numButCat)

catButCar = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

# Tip olarak category veya object olan ve 20'den büyük eşsiz sınıf sayısı  olan sütunların listesi
print(catButCar)
catCols = catCols + numButCat
print(catCols)

catCols = [col for col in catCols if col not in catButCar]
print(catCols)

print(df[catCols].nunique())

print([col for col in df.columns if col not in catCols])


def catSummary(dataFrame, colName):
    print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                        "Ratio": 100 * dataFrame[colName].value_counts() / len(dataFrame)}))
    print("##########################################")


catSummary(df, "sex")
for col in catCols:
    catSummary(df, "sex")


def catSummary(dataFrame, colName, plot=False):
    print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                        "Ratio": 100 * dataFrame[colName].value_counts() / len(dataFrame)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataFrame[colName], data=dataFrame)
        plt.show(block=True)


catSummary(df, "sex", plot=True)

for col in catCols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        catSummary(df, col, plot=True)

df["adult_male"].astype(int)
print(df["adult_male"])

for col in catCols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        catSummary(df, col, plot=True)
    else:
        catSummary(df, col, plot=True)


def catSummary(dataFrame, colName, plot=False):
    if dataFrame[colName].dtypes == "bool":
        dataFrame[colName] = dataFrame[colName].astype(int)
        print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                            "Ratio": 100 * dataFrame[colName].value_counts() / len(dataFrame)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataFrame[colName], data=dataFrame)
            plt.show(block=True)
    else:
        print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                            "Ratio": 100 * dataFrame[colName].value_counts() / len(dataFrame)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataFrame[colName], data=dataFrame)
            plt.show(block=True)


catSummary(df, "sex", plot=False)



# 3. Analysis of Numerical Variables

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/advertising.csv")
df = sns.load_dataset("titanic")
print(df.head())

catCols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
print(catCols)  # Tip olarak object, category veya bool olan sütunların listesi

numButCat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
# Tip olarak int veya float olan sütunların listesi ve 10'dan küçük nunique değerleri olan sütunların listesis
print(numButCat)

catButCar = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

# Tip olarak category veya object olan ve 20'den büyük eşsiz sınıf sayısı  olan sütunların listesi
print(catButCar)
catCols = catCols + numButCat
print(catCols)

catCols = [col for col in catCols if col not in catButCar]
print(catCols)
numCols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
print(numCols)
numCols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
numCols = [col for col in numCols if col not in catCols]
print(numCols)


def numSummary(dataframe, numericalCol):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numericalCol].describe(quantiles).T)


numSummary(df, "age")

for col in numCols:
    numSummary(df, col)


def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numericalCol].describe(quantiles).T)
    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)


numSummary(df, "age", plot=True)

for col in numCols:
    numSummary(df, col, plot=True)


# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi

def grabColNames(dataFrame, catTh=10, carTh=30):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.
    """
    catCols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
    numButCat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    catButCar = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    catCols = catCols + numButCat
    catCols = [col for col in catCols if col not in catButCar]
    numCols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    numCols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    numCols = [col for col in numCols if col not in catCols]
    print(f"Observations: {dataFrame.shape[0]}")
    print(f"Variables: {dataFrame.shape[1]}")
    print(f'cat_cols: {len(catCols)}')
    print(f'num_cols: {len(numCols)}')
    print(f'cat_but_car: {len(catButCar)}')
    print(f'num_but_cat: {len(numButCat)}')

    return catCols, numCols, catButCar


catCols, numCols, catButCar = grabColNames(df)


def catSummary(dataFrame, colName):
    print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                        "Ratio": 100 * dataFrame[colName].value_counts() / len(dataFrame)}))
    print("##########################################")


catSummary(df, "sex")
for col in catCols:
    catSummary(df, col)


def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numericalCol].describe(quantiles).T)
    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)

for col in numCols:
    numSummary(df, col, plot=True)

# bonus
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

catCols, numCols, catButCar = grabColNames(df)
def catSummary(dataframe, colName, plot=False):
    print(pd.DataFrame({colName: dataframe[colName].value_counts(),
                        "Ratio": 100 * dataframe[colName].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[colName], data=dataframe)
        plt.show(block=True)


for col in catCols:
    catSummary(df, col, plot=True)

for col in numCols:
    numSummary(df, col, plot=True)



# 4. Analysis of Target Variable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/advertising.csv")
df = sns.load_dataset("titanic")
print(df.head())

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
def catSummary(dataframe, colName, plot=False):
    print(pd.DataFrame({colName: dataframe[colName].value_counts(),
                        "Ratio": 100 * dataframe[colName].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[colName], data=dataframe)
        plt.show(block=True)

def grabColNames(dataFrame, catTh=10, carTh=30):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.
    """
    catCols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
    numButCat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    catButCar = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    catCols = catCols + numButCat
    catCols = [col for col in catCols if col not in catButCar]
    numCols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    numCols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    numCols = [col for col in numCols if col not in catCols]
    print(f"Observations: {dataFrame.shape[0]}")
    print(f"Variables: {dataFrame.shape[1]}")
    print(f'cat_cols: {len(catCols)}')
    print(f'num_cols: {len(numCols)}')
    print(f'cat_but_car: {len(catButCar)}')
    print(f'num_but_cat: {len(numButCat)}')

    return catCols, numCols, catButCar

catCols, numCols, catButCar = grabColNames(df)

df["survived"].value_counts()
catSummary(df, "survived")

# Hedef Değişkenin Kategorik Değişkenler ile Analizi

def targetSummaryWithCat(dataframe, target, categoricalCol):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categoricalCol)[target].mean()}), end="\n\n\n")

targetSummaryWithCat(df, "survived", "pclass")

for col in catCols:
    targetSummaryWithCat(df, "survived", col)
    print("##########################################")

# Hedef Değişkenin Sayısal Değişkenler ile Analizi
def targetSummaryWithCat(dataframe, target, numericalCol):
    print(dataframe.groupby(target).agg({numericalCol: "mean"}), end="\n\n\n")

targetSummaryWithCat(df, "survived", "age")

for col in numCols:
    targetSummaryWithCat(df, "survived", col)



# 5. Korelasyon Analizi (Analysis of Correlation)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
print(df.head())

numCols = [col for col in df.columns if df[col].dtype in [int, float]]
print(numCols)

corr = df[numCols].corr()  # korelasyon matrisi oluşturma
print(corr)
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


# Yüksek Korelasyonlu Değişkenlerin Silinmesi

corMatrix = df.corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

upperTriangleMatrix = corMatrix.where(np.triu(np.ones(corMatrix.shape), k=1).astype(bool))
drop_list = [col for col in upperTriangleMatrix.columns if any(upperTriangleMatrix[col]>0.90) ]
corMatrix[drop_list]
df.drop(drop_list, axis=1)
print(df.head())

def highCorrelatedCols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corMatrix = corr.abs()
    upperTriangleMatrix = corMatrix.where(np.triu(np.ones(corMatrix.shape), k=1).astype(bool))
    dropList = [col for col in upperTriangleMatrix.columns if any(upperTriangleMatrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return dropList

highCorrelatedCols(df)
drop_list = highCorrelatedCols(df, plot=True)
df.drop(drop_list, axis=1)
highCorrelatedCols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
print(len(df.columns))
print(df.head())

drop_list = highCorrelatedCols(df, plot=True)

print(len(df.drop(drop_list, axis=1).columns))