# VERİ YAPILARI (DATA STRUCTURES)

# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set


# Data Structeres (Veri Yapıları)
# Sayılar : integer => Tam sayıları ifade eder.
x = 46
print(x)
print(type(x))

#Sayılar : float => ondalıklı sayıları ifade eder.
y = 10.3
print(y)
print(type(y))

#Sayılar: complex
z = 2j+1
print(z)
print(type(z))

#String => metinsel ifadelerdir. Tırnak içerisinde yazılır.
a = "Hello ai era"
print(a)
print(type(a))

# Boolean => true ya da false değeri döndüren mantıksal ifadelerdir.
print(5 == 4)
print(3 == 2)

#Liste
b = ["a","b","c"]
print(type(b))

#Dictionary(Sözlük) => key ve value değerleri alır
c = {"name" : "Berkcan","Age":23}
print(c)
print(type(c))

#Tuple
d = ("python","ml","ds")
print(d)
print(type(d))

#Set => kümelere benzer
e = {"python","ml","ds"}
print(e)
print(type(e))


#########################################################

#Sayılar(numbers) => int, float, complex

a = 5
b = 10.5

print(a * 3) # çarpma işlemi yapar
print(a/ 7)
print(a * b /10)
print(a ** 2) # üs almayı sağlar.


# Tip Dönüşümü

# int() => float değeri integer değere çevirir.
# float() => integer değeri float değere çevirir.

print(int(b))
print(float(a))
print(int(a * b / 10))

c = a * b / 10
print(int(c))


################################################

# Karakter Dizileri(String)
print("John")
print("John")

name = "miuul"

# Çok Satırlı Karakter Dizileri => 3 tane tek tırnak ya da 3 çift tırnak içerisinde yazılır.

print(
    """
    Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool
    """
)

longStr = """Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""
print(longStr)

# Karakter Dizilerinin Elemanlarına Erişmek
# Python programlama dilinde index 0'dan başlar.
# İstenilen indexe erişimek için [] içerisinde yazılır.
print(name[0])
print(name[3])

# Karakter Dizilerinde Slice İşlemi => belirlenen indexten istenen indexe kadar almayı sağlar.
print(name[0:2])
print(longStr[0:10])

# String içerisinde karakter sorgulama
# in operatörü ile yapılır.
# Büyük küçük harf duyarlıdır.
print("Veri" in longStr)
print("veri" in longStr)
print("bool" in longStr)

#String Metotları

dir(str)

# len metodu => string değerin kaç elemandan oluştuğun verir.
print(len(name))
print(len("vahitkeskin"))
print(len("miuul"))

# upper() & lower(): küçük-büyük dönüşümleri
print(name.upper())
print(name.lower())

# replace: karakter değiştirir
hi = "Hello AI Era"
hi.replace("l", "p")
print(hi)

# split: Belli bir boşluğa ya da karaktere göre böler
print(hi.split())

# strip: Baştan ve sondan boşlukları kırpar.
print(" ofofo ".strip())
print("ofofo".strip("o"))

# capitalize : İlk harfi büyütür.
print("foo".capitalize())


###############################################################

# Liste (List)

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.
# - [] içerisine yazılır ve , ile ayrılır.
# - Her tür veri tipi içeride olabilir.

notes = [1,2,3,4]
print(type(notes))
names = ["a", "b", "v", "d"]
not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]
print(not_nam[0])
print(not_nam[5])
print(not_nam[6])
print(not_nam[6][1])

notes[0] = 99
print(notes)
print(not_nam[0:3])

#Liste Metodları (List Methods)
#len() => kaç elemandan oluştuğu bilgisini verir.
print(len(notes))
print(len(not_nam))

# append: Listenin sonuna eleman ekler.
notes.append(100)
print(notes)

# pop: indexe göre silme işlemi yapar.
notes.pop(1)
print(notes)

# insert: indexe göre eleman ekler
notes.insert(2,5)
print(notes)


###############################################################

# Dictionary (Sözlük)

# Değiştirilebilir.
# Normal şartlarda sırasızdır. Ama 3.7 versiyonuyla birlikte sıralı özelliği kazanmıştır.
# Her veri tipini içerisinde barındırır.
# {} içerisine yazılır. {key:value}
# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}
print("REG")

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

print(dictionary["CART"])

# Key Sorgulama

print("YSA" in dictionary)

# Key'e Göre Value'ya Erişmek => [""] ya da get metoduyla erişiriz.


print(dictionary["REG"])
print(dictionary.get("REG"))

# Value Değiştirmek

dictionary["REG"] = ["YSA", 10]
print(dictionary)


# Tüm Key'lere Erişmek

print(dictionary.keys())

# Tüm Value'lara Erişmek

print(dictionary.values())


# Tüm Çiftleri Tuple Halinde Listeye Çevirme
print(dictionary.items())

# Key-Value Değerini Güncellemek

dictionary.update({"REG": 11})


# Yeni Key-Value Eklemek

dictionary.update({"RF": 10})

print(dictionary)


############################################################

# Tuple (Demet)

# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.
# - () içerisine yazılır

t = ("john", "mark", 1, 2)
print(type(t))

print(t[0])
print(t[0:3])


t = list(t)
print(t)
t[0] = 99
t = tuple(t)
print(t)


############################################################

# Set(Küme)

# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.

# difference(): İki kümenin farkı


set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
print(set1.difference(set2))
print(set2.difference(set1))

# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
print(set1.symmetric_difference(set2))
print(set2.symmetric_difference(set1))

# intersection(): İki kümenin kesişimi
print(set1.intersection(set2))

# union(): İki kümenin birleşimi
print(set1.union(set2))

# isdisjoint(): İki kümenin kesişimi boş mu?
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])
print(set1.isdisjoint(set2))

# issubset(): Bir küme diğer kümenin alt kümesi mi?
print(set1.issubset(set2))
print(set2.issubset(set1))

# issuperset(): Bir küme diğer kümeyi kapsıyor mu?
print(set1.issuperset(set2))
print(set2.issuperset(set1))
