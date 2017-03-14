from readClass import *
import numpy as np
import pandas as pd

data = np.loadtxt('./insuranceCompany_Data/ticdata2000.txt')

X = InsuranceCompanyData(data)
print("InsuranceCompanyData\n")
print("# of observations:\t", len(X.data))
print("# of classes:\t\t", len(X.data[0]))
print("# of groups:\t\t", len(X.classGroup))
print()
[0, 1, 2, 3, 8, 10]
# df = X.stat(8).values
# print(df[1][0])
for i in range(len(X.classGroup)):
    X.stat(i, True)
X.summary(1, 4)
X.astat(0)
X.astat(1)
X.astat(2)
X.astat(3)
X.astat(9)
X.astat(17)
X.astat(41)
X.astat(46)
X.astat(58)
X.astat(85)
X.astat(79)

# Y = X.data[12, 5:42]
# for i in X.data:
# counts = []
# inc = 0
# df = X.data
# for i in range(len(df)):
#     print(i)
#     count = 0
#     dlist = []
#     Y = df[i, 5:42]
#     for j in range(len(df)):
#         if (df[j, 5:42] == Y).all():
#             count+=1
#             dlist.append(j)
#     for d in dlist:
#         np.delete(df, d, 0)
#     counts.append(count)
# print(counts)
