import numpy as np

text = "2,3,4;5,6,7"
rows = text.split(";")
matrix = []
for i in rows:
    rowElems = i.split(",")
    nums = []
    for j in rowElems:
        nums.append(int(j))
    matrix.append(nums)

print(matrix)