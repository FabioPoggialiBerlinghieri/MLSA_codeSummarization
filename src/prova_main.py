import ast
import codeTokenizer as tc

snippets = [
    # 1. Variabili
    """
x = 10
y = 20
result = x + y
print(result)
""",

    # 2. If / Else
    """
x = 15
if x > 10:
    print("Maggiore di 10")
else:
    print("Minore o uguale a 10")
""",

    # 3. For
    """
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    print(n)
""",

    # 4. While
    """
x = 0
while x < 5:
    print(x)
    x += 1
""",

    # 5. List comprehension
    """
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(squares)
""",

    # 6. Filter
    """
numbers = [1, 2, 3, 4, 5, 6]
even = [x for x in numbers if x % 2 == 0]
print(even)
""",

    # 7. String methods
    """
text = "hello world"
text = text.upper()
words = text.split()
print(words)
""",

    # 8. f-string
    """
name = "Pier"
age = 25
message = f"{name} ha {age} anni"
print(message)
""",

    # 9. Dictionary
    """
person = {"name": "Pier", "age": 25}
print(person["name"])
person["age"] += 1
""",

    # 10. Dictionary comprehension
    """
numbers = [1, 2, 3, 4]
squares = {x: x**2 for x in numbers}
print(squares)
""",

    # 11. Funzione
    """
def square(x):
    result = x ** 2
    return result

print(square(5))
""",

    # 12. Funzione con default
    """
def greet(name="World"):
    message = f"Hello {name}"
    return message

print(greet())
""",

    # 13. Lambda
    """
numbers = [1, 2, 3, 4]
square = lambda x: x ** 2
result = [square(x) for x in numbers]
print(result)
""",

    # 14. Map
    """
numbers = [1, 2, 3, 4]
squares = map(lambda x: x**2, numbers)
squares = list(squares)
print(squares)
""",

    # 15. Enumerate
    """
names = ["Anna", "Marco", "Luca"]
for i, name in enumerate(names):
    print(i, name)
""",

    # 16. Zip
    """
names = ["Anna", "Marco", "Luca"]
ages = [20, 25, 30]
for name, age in zip(names, ages):
    print(name, age)
""",

    # 17. Try / Except
    """
try:
    x = int(input("Numero: "))
    print(10 / x)
except ValueError:
    print("Input non valido")
""",

    # 18. Set
    """
numbers = [1, 2, 2, 3, 3, 4]
unique = set(numbers)
print(unique)
print(len(unique))
""",

    # 19. Sorting
    """
numbers = [5, 2, 8, 1, 3]
numbers.sort()
print(numbers)
numbers.sort(reverse=True)
print(numbers)
""",

    # 20. Sorted con key
    """
names = ["Anna", "Alessandro", "Luca"]
result = sorted(names, key=len)
print(result)
""",

    # 21. File reading
    """
with open("data.txt", "r") as file:
    content = file.read()

print(content)
""",

    # 22. File writing
    """
numbers = [1, 2, 3, 4]
with open("data.txt", "w") as file:
    for n in numbers:
        file.write(f"{n}\\n")
""",

    # 23. JSON
    """
import json

data = {"name": "Pier", "age": 25}
text = json.dumps(data)
print(text)
""",

    # 24. Random
    """
import random

numbers = [1, 2, 3, 4, 5]
choice = random.choice(numbers)
print(choice)
""",

    # 25. Counter
    """
from collections import Counter

letters = "banana"
counts = Counter(letters)
print(counts)
""",

    # 26. NumPy array
    """
import numpy as np

x = np.array([1, 2, 3, 4])
result = x * 2
print(result)
""",

    # 27. NumPy mean
    """
import numpy as np

data = np.array([10, 20, 30, 40])
mean = np.mean(data)
print(mean)
""",

    # 28. Pandas DataFrame
    """
import pandas as pd

df = pd.DataFrame({"age": [20, 25, 30]})
print(df)
""",

    # 29. Pandas filtering
    """
import pandas as pd

df = pd.DataFrame({"age": [20, 25, 30]})
result = df[df["age"] > 20]
print(result)
""",

    # 30. Pandas groupby
    """
import pandas as pd

df = pd.DataFrame({"city": ["Rome", "Rome", "Milan"],
                   "age": [20, 30, 25]})
result = df.groupby("city")["age"].mean()
print(result)
""",

"""
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
""",

"""
class Student(Person):
    def __init__(self, name, age, grades):
        super().__init__(name, age)
        self.grades = grades
    def average(self):
        return sum(self.grades) / len(self.grades)
"""
]

voc = []
for snip in snippets:
    voc.extend(tc.ASTToSBT().parse(ast.parse(snip)).split())
occ_dict = dict.fromkeys(voc, 0)
for v in voc:
    occ_dict[v] += 1

occ_dict = dict(sorted(occ_dict.items(), key=lambda x: x[1], reverse=True))
# todo eliminare vocaboli in eccesso (oltre 30 000)
values = range(9, len(occ_dict) + 9)
voc = dict(zip(occ_dict.keys(), values))
special_tokens = {'Var': 0, 'Constant': 1, 'Function': 2, 'Class': 3, 'Argument': 4, 'Attr': 5, 'Kwarg':6, 'Import':7, 'UNK': 8}
voc = special_tokens | voc

code1 = """
class Prova:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def square(x):
        return x ** 2
"""

codeTokenizer = tc.CodeTokenizer(voc)
result = codeTokenizer.tokenize(code1)
print(result)
for res in result:
    chiave = next((k for k, v in voc.items() if v == res), None)
    print(chiave)
