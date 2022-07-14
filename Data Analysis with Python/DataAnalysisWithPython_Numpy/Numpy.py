# NUMPY

# Neden Numpy? (Why Numpy)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)


# Why Numpy?

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

ab

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b


# Creating Numpy Arrays

import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))


# Attibutes of Numpy Arrays

import numpy as np

# ndim: size number (boyut sayısı)
# shape: size information (boyut bilgisi)
# size: total number of elements (toplam eleman sayısı)
# dtype: array data type (dizi veri tipi)

a = np.random.randint(10, size=5)

a.ndim
a.shape
a.size
a.dtype


# Reshaping (Yeniden Şekillendirme)

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)


# Index Selection

a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999
a

m = np.random.randint(10, size=(3, 5))
m
m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 999
m

m[2, 3] = 2.9  # Numpy, sabit tipli array dir. Aynı tip bilgisi saklar.
m

m[:, 0]
m[1, :]
m[0:2, 0:3]


# Fancy Index

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3, 6]

v[catch]


# Conditions on Numpy (Numpy'da Koşullu İşlemler)

import numpy as np

v = np.array([1, 2, 3, 4, 5])

# for döngüsü ile;
ab = []

for i in v:
    if i < 3:
        ab.append(i)

ab

# Numpy ile;
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]


# Mathematical Operations

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

# NumPy ile İki Bilinmeyenli Denklem Çöüzümü

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)