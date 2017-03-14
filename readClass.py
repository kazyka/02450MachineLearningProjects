import numpy as np
import pandas as pd
from statistics import mode

class InsuranceCompanyData:
    def __init__(self, data):
        self.data = data
        self.classGroup = self.readClass()
        self.L = self.readL()
    def readClass(self):
        classes = []
        classGroup = []
        with open("insuranceCompany_Data/class") as f:
            content = f.readlines()
            for line in content:
                if(line.isspace()):
                    classGroup.append(classes)
                    classes = []
                    continue
                split = line.split(" ")
                classes.append((int(split[0]), split[1],  " ".join(split[2:]).strip("\n")))
            # append last group
            classGroup.append(classes)
        return classGroup
    def readL(self):
        def s(num):
            L = []
            with open("insuranceCompany_Data/L"+str(num)) as l:
                content = l.readlines()
                for line in content:
                    split = line.split(" ")
                    L.append((int(split[0]),  " ".join(split[1:]).strip("\n")))
            return L
        Ls = []
        for i in range(5):
            Ls.append(s(i))
        return Ls
    def classGroupL(self, classGroup):
        L = self.L
        groupNum = classGroup[0][0]
        if groupNum == 1:
            l = L[0]
        elif groupNum == 4:
            l = L[1]
        elif groupNum == 5:
            l = L[2]
        elif 6 <= groupNum and groupNum <= 43:
            l = L[3]
        elif 44 <= groupNum and groupNum <= 64:
            l = L[4]
        else:
            print("Numerical")
            return
        for i in l:
            print(str(i[0])+" "+i[1])
    def classGroupStat(self, classGroup, code=False):
        data = self.data
        classDict = {}
        modeDict = {}
        for classes in classGroup:
            colName = classes[2] if not code else " ".join([classes[1],classes[2]])
            classDict[colName] = data[:, classes[0]-1]
            modeDict[colName] = mode(data[:, classes[0]-1])
        stat = pd.DataFrame(classDict).describe()
        return stat.append(pd.DataFrame(modeDict, index=['mode']))
    def stat(self, groupNum, showMinMax=False):
        group = self.classGroup[groupNum]
        attr = (group[0][0]-1, group[0][0]+len(group)-1)
        self.classGroupL(group)
        # print(self.data[0, attr[0]:attr[1]])
        df = self.classGroupStat(group)
        df = df.truncate() if showMinMax else df.truncate(after='std')
        print(df)
        colName = df.columns.values
        maxMeanInGroup, maxStdInGroup = np.argmax(df.values[1]), np.argmax(df.values[2])
        minMeanInGroup, minStdInGroup = np.argmin(df.values[1]), np.argmin(df.values[2])
        print(maxMeanInGroup, maxStdInGroup, minMeanInGroup, minStdInGroup)
        print("Max mean:\t", colName[maxMeanInGroup][:29], "\t\t", df.values[1][maxMeanInGroup])
        print("Min mean:\t", colName[minMeanInGroup][:29], "\t\t", df.values[1][minMeanInGroup])
        print("Max std:\t", colName[maxStdInGroup][:29], "\t\t", df.values[2][maxStdInGroup])
        print("Min std:\t", colName[minStdInGroup][:29], "\t\t", df.values[2][minStdInGroup])
    def astat(self, aNum):
        df = pd.DataFrame({"att":self.data[:, aNum]}).describe()
        df = df.append(pd.DataFrame({"att":mode(self.data[:, aNum])}, index=['mode']))
        val = ['%.3f' % i for j in df.values for i in j]
        print(" & ".join([val[1],val[2],val[3],val[7],val[8]]), "\\\\")
        return df
    def summary(self, start=0, end=86):
        mean = []
        std = []
        for i in self.classGroup:
            df = self.classGroupStat(i)
            mean.append(j for j in df.values[1])
            std.append(df.values[2])
        mean, std = [i for j in mean for i in j], [i for j in std for i in j]
        mean, std= mean[start:end], std[start:end]
        print("mean over each attribute:\n", "max\t", np.argmax(mean, 0), np.amax(mean), "\n min\t", np.argmin(mean)+1, np.amin(mean))
        print("std over each attribute:\n", "max\t", np.argmax(std), np.amax(std), "\n min\t", np.argmin(std)+1, np.amin(std))
