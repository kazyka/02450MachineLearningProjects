from resolve_path import *
from readClass import *
from sys import platform
from subprocess import run
from writeapriorifile import WriteAprioriFile
import numpy as np
import re
import os
import time

if platform == "linux" or platform == "linux2":
	ext = ''  # Linux
	dir_sep = '/'
	tool_path = '..{0}..{0}Toolbox{0}02450Toolbox_Python{0}Tools'.format(dir_sep)
elif platform == "darwin":
	ext = 'MAC'  # OS X
	dir_sep = '/'
	tool_path = '..{0}..{0}02450Toolbox_Python{0}Tools'.format(dir_sep)
elif platform == "win32":
	ext = '.exe'  # Windows
	dir_sep = '\\'
else:
    raise NotImplementedError()

filename = 'AprioriFileProject03_N{0}.txt'.format(X.shape[0])
minSup = 80
minConf = 100
maxRule = 4

att = np.array(list(range(43,M)))
att = np.concatenate(([0,1,2,3], att), axis=0)

X = X[:,:]
L = np.array([classnum[i] for i in att])
def lookup(idx, verbose=True):
	i = 0
	idx+=1
	while idx - L[i] > 0:
		idx-=L[i]
		i+=1
	if i >= 5: idx-=1
	if verbose: print("{0}: {1}".format(classname[att[i]], idx))
	return idx
if not os.path.isfile(filename):
	Xnew = np.zeros((X.shape[0], L.sum()))
	for idx, obs in enumerate(X):
		base = 0
		for i_idx, i in enumerate(att):
			a = base + obs[i] - 1
			if i >= 5: a += 1
			Xnew[idx][int(a)] = 1
			base += L[i_idx]
	WriteAprioriFile(Xnew,filename=filename)
# Run Apriori Algorithm
print('Mining for frequent itemsets by the Apriori algorithm')
status1 = run('{4}{3}apriori{0} -f"," -s{1} -v"[Sup. %S]" {2} apriori_temp1.txt'
              .format(ext, minSup, filename, dir_sep, tool_path), shell=True)

if status1.returncode != 0:
    print('An error occurred while calling apriori, a likely cause is that minSup was set to high such that no '
          'frequent itemsets were generated or spaces are included in the path to the apriori files.')
    exit()
if minConf > 0:
    print('Mining for associations by the Apriori algorithm')
    status2 = run('{6}{5}apriori{0} -tr -f"," -n{1} -c{2} -s{3} -v"[Conf. %C,Sup. %S]" {4} apriori_temp2.txt'
                  .format(ext, maxRule, minConf, minSup, filename, dir_sep, tool_path), shell=True)

    if status2.returncode != 0:
        print('An error occurred while calling apriori')
        exit()
print('Frequent itemsets mining done, extracting results')

# Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
f = open('apriori_temp1.txt', 'r')
lines = f.readlines()
f.close()
# Extract Frequent Itemsets
FrequentItemsets = [''] * len(lines)
sup = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    FrequentItemsets[i] = line[0:-1]
    sup[i] = re.findall(' [-+]?\d*\.\d+|\d+]', line)[0][1:-1]
os.remove('apriori_temp1.txt')
FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup, axis=0).ravel()]
FrequentItemsetsSorted.reverse()

null_f_item = 0
f_item = []
print('Frequent itemsets [{0}]:'.format(len(FrequentItemsetsSorted)))
for i, item in enumerate(FrequentItemsetsSorted):
	s = re.findall(r'\b\d+\.*\d+\b', item)
	flag = 0
	for f_it in s[:-1]:
		if lookup(int(f_it), False) > 0: flag = 1
	if flag == 0: null_f_item+=1
	else: f_item.append(item)

print('{0} out of {1} rules are null'.format(null_f_item, len(FrequentItemsetsSorted)))
