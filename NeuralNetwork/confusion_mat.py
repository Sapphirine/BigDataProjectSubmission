# Big Data Analytics
# Fall 2015
#
# Jie Yuan
# Ziyu He
# Yubin Shen

# This script prints a confusion matrix given a list of
# actual and predicted classes.
num_classes = 10
mat = [[0] * num_classes for i  in range(num_classes)]
with open("subset_output.csv") as f:
    for line in f:
        line = line.strip().split(",")
        row = int(line[0])
        col = int(line[1])
        mat[row][col] += 1
for row in mat:
    for num in row:
        print str(num) + ",",
    print
