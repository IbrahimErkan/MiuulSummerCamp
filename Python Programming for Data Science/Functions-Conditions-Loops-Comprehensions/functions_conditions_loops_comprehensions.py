# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS

# - Fonksiyonlar (Functions)
# - Koşullar - (Conditions)
# - Döngüler (Loops)
# - Comprehensions


# FONKSİYONLAR (FUNCTIONS)

# fonksiyon : belirli görevleri yerine getiren kod parçalarıdır.

# Fonksiyon Okuryazarlığı

print("a", "b")
print("a", "b", sep="--")

print("a")


# Fonksiyon Tanımlama


def calculate(x):
    print(x * 2)


calculate(5)


# İki argümanlı/parametreli bir fonksiyon tanımlayalım.

def summer(arg1, arg2):
    print(arg1 + arg2)


summer(5, 9)


# Docstring

def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    """
    print(arg1 + arg2)


summer(1, 2)


# Fonksiyonların Statement/Body Bölümü

# def function_name(parameters/arguments):
#     statements (function body)

def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")


say_hi()


def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 5)

# girilen değerleri bir liste içinde saklayacak fonksiyon.

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 5)
add_element(180, 81)


# Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)

def divide(a, b):
    print(a / b)


divide(1, 2)


def divide(a, b=1):
    print(a / b)


divide(1)


def say_hi(string="Merhaba"):
    print(string)
    print("Hi")
    print("Hello")


say_hi()

# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?

# varm, moisture, charge

(57 + 25) / 80
(24 + 45) / 70
(67 + 82) / 80


# DRY

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 78)


# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78) * 10


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output


calculate(42, 25, 64)
varm, moisture, charge, output = calculate(42, 25, 64)


# Fonksiyon içerisinden Fonksiyon Çağırmak

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(varm, moisture, charge, a, b):
    print(calculate(varm, moisture, charge))
    b = standardization(a, b)
    print(b * 10)


all_calculation(1, 3, 5, 19, 12)

# Local & Global Değişkenler (Local & Global Variables)

list_store = [1, 2]


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(5, 9)

# KOŞULLAR (CONDITIONS)


# if
if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11

if number == 11:
    print("number is 10")


def number_check(number):
    if number == 10:
        print("number is 10")


number_check(10)
number_check(12)


# else

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")


number_check(12)


# elif

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(10)

# DÖNGÜLER (LOOPS)

# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 20 / 100 + salary))

for salary in salaries:
    print(int(salary * 30 / 100 + salary))


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 10))

salaries2 = [10700, 25000, 30400, 40300, 50200]
for salary in salaries2:
    print(new_salary(salary, 15))

salaries = [1000, 2000, 3000, 4000, 5000]
for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 15))
    else:
        print(new_salary(salary, 20))


# Uygulama - Mülakat Sorusu

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
    new_string = ""
    # girilen string'in indexlerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir.
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçğk harfe çevir.
        else:
            new_string += string[string_index].lower()
    print(new_string)


alternating("miuul")

# break & continue & while

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# while

number = 1
while number < 5:
    print(number)
    number += 1

# Enumerate: Otomatik Counter/Indexer ile for loop

students = ["John", "Mark", "Venessa", "Mariam"]

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

A, B

# Uygulama- Mülakat Sorusu

# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


st = divide_students(students)
st[0]
st[1]


# alternating fonksiyonunun enumerate ile yazılması

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john i am learning python")

# Zip

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))


# lambda, map, filter, ruduce

def summer(a, b):
    return a + b


summer(1, 3) * 9

new_sum = lambda a, b: a + b
new_sum(4, 5)

# map
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

# del new_sum
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2, salaries))

# filter
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# reduce
from functools import reduce

list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)

# COMPREHENSIONS

# List Comprehensions

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

null_list

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]
[salary * 2 for salary in salaries if salary < 3000]
[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

# Dict Comprehension

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}
{k.upper(): v for (k, v) in dictionary.items()}
{k.upper(): v ** 2 for (k, v) in dictionary.items()}

# Uygulama - Mülakat Sorusu

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir.

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

new_dict

{n: n ** 2 for n in numbers if n % 2 == 0}

# List & Dict Comprehension Uygulamalar

# Bir Veri Setindeki Değişken İsimleri Değiştirmek

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']


import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []
for col in df.columns:
    A.append(col.upper())

df.columns = A
df.columns

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]
df.columns


# İsminde "INS" olan değişkenlerin başına Flag diğerlerine NO_FLAG eklemek istiyoruz:

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]
df.columns


# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns


num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

soz

#kısa yol
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)


##############################################################


