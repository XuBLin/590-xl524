import sys
import numpy as np

num = sys.stdin.readline().strip(' ')
num = int(num)
string = sys.stdin.readline().strip(' ')
array = string[:num]
for i in range(num):
    array[i] = int(array[i])
matrix = np.zeros((num, num))

for i in range(0, num):
    if array[num-1] % array[num] == 0:
        matrix[i][num-1] = 2
    else:
        matrix[i][num-1] = 1

# 6
# 0001 0002 0001 0002 0003 0002
