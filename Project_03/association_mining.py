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
minSup = 58
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
def lookupstr(idx):
	i = 0
	idx+=1
	while idx - L[i] > 0:
		idx-=L[i]
		i+=1
	if i >= 5: idx-=1
	return "{0}: {1}".format(classname[att[i]], idx)
def interpret(rule):
	s = re.findall(r'\b\d+\.*\d+\b', rule)
	print("{0}\n=> {1} [ C: {2} S: {3}]".format("\n".join([lookupstr(int(a)) for a in s[1:-2]]), lookupstr(int(s[0])), s[-2], s[-1]))
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

if minConf > 0:
    print('Mining for associations by the Apriori algorithm')
    status2 = run('{6}{5}apriori{0} -tr -f"," -n{1} -c{2} -s{3} -v"[Conf. %C,Sup. %S]" {4} apriori_temp2.txt'
                  .format(ext, maxRule, minConf, minSup, filename, dir_sep, tool_path), shell=True)

    if status2.returncode != 0:
        print('An error occurred while calling apriori')
        exit()
print('Association mining done, extracting results')

f = open('apriori_temp2.txt', 'r')
lines = f.readlines()
f.close()
# Extract Association rules
AssocRules = [''] * len(lines)
conf = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    AssocRules[i] = line[0:-1]
    conf[i] = re.findall(' [-+]?\d*\.\d+|\d+,', line)[0][1:-1]
os.remove('apriori_temp2.txt')
AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf, axis=0).ravel()]
AssocRulesSorted.reverse()

null_rule = 0
g_rule = []
omit_attr = [41]
print('Association rules [{0}]:'.format(len(AssocRulesSorted)))
for i, item in enumerate(AssocRulesSorted):
	s = re.findall(r'\b\d+\.*\d+\b', item)
	flag = 0
	for a in s[:-2]:
		if int(a) in omit_attr: continue
		if lookup(int(a), False) > 0: flag = 1
	if flag == 0: null_rule+=1
	else: g_rule.append(item)
print('{0} out of {1} rules are null (omitting {2})'.format(null_rule, len(AssocRulesSorted), omit_attr))
