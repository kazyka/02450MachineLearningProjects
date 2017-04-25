# exercise 10.2.1
from resolve_path import *

from writeapriorifile import WriteAprioriFile
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import OneHotEncoder

def runApriori(filename,minSup=40,minConf=100,maxRule=4):
	from subprocess import run
	import re
	import os

	# Run Apriori Algorithm
	#print('Mining for frequent itemsets by the Apriori algorithm')
	if os.uname().sysname.lower() == 'linux':
	    status1 = run('../../Toolbox/02450Toolbox_Python/Tools/apriori -f -s{0} -v"[Sup. %S]" {1} apriori_temp1.txt'.format(minSup, filename),shell=True)
	else:
	    status1 = run('../../Toolbox/02450Toolbox_Python/Tools/apriori.exe -f -s{0} -v"[Sup. %S]" {1} apriori_temp1.txt'.format(minSup, filename),shell=True)

	if status1.returncode!=0:
	    print('An error occured while calling apriori, a likely cause is that minSup was set to high such that no frequent itemsets were generated or spaces are included in the path to the apriori files.')
	    exit()
	if minConf>0:
	    #print('Mining for associations by the Apriori algorithm')
	    
	    if os.uname().sysname.lower() == 'linux':
	        status2 = run('../../Toolbox/02450Toolbox_Python/Tools/apriori -tr -f"," -n{0} -c{1} -s{2} -v"[Conf. %C,Sup. %S]" {3} apriori_temp2.txt'.format(maxRule, minConf, minSup, filename),shell=True)
	    else:
	        status2 = run('../../Toolbox/02450Toolbox_Python/Tools/apriori.exe -tr -f"," -n{0} -c{1} -s{2} -v"[Conf. %C,Sup. %S]" {3} apriori_temp2.txt'.format(maxRule, minConf, minSup, filename),shell=True)
	    
	    if status2.returncode!=0:
	        print('An error occured while calling apriori')
	        exit()
	#print('Apriori analysis done, extracting results')


	# Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
	f = open('apriori_temp1.txt','r')
	lines = f.readlines()
	f.close()
	# Extract Frequent Itemsets
	FrequentItemsets = ['']*len(lines)
	sup = np.zeros((len(lines),1))
	for i,line in enumerate(lines):
	    FrequentItemsets[i] = line[0:-1]
	    sup[i] = re.findall(' [-+]?\d*\.\d+|\d+]', line)[0][1:-1]
	os.remove('apriori_temp1.txt')
	    
	# Read the file
	f = open('apriori_temp2.txt','r')
	lines = f.readlines()
	f.close()
	# Extract Association rules
	AssocRules = ['']*len(lines)
	conf = np.zeros((len(lines),1))
	for i,line in enumerate(lines):
	    AssocRules[i] = line[0:-1]
	    conf[i] = re.findall(' [-+]?\d*\.\d+|\d+,', line)[0][1:-1]
	os.remove('apriori_temp2.txt')    

	# sort (FrequentItemsets by support value, AssocRules by confidence value)
	AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf,axis=0).ravel()]
	AssocRulesSorted.reverse()
	FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup,axis=0).ravel()]
	FrequentItemsetsSorted.reverse()
	    
	# Print the results
	import time; time.sleep(.5)    
	'''print('\n')
	print('RESULTS:\n')
	print('Frequent itemsets:')
	for i,item in enumerate(FrequentItemsetsSorted):
	    print('Item: {0}'.format(item))
	print('\n')
	print('Association rules:')
	for i,item in enumerate(AssocRulesSorted):
	    print('Rule: {0}'.format(item))'''


# Load Matlab data file and extract variables of interest
enc = OneHotEncoder()
enc.fit(X)
Xnew = enc.transform(X).toarray()

WriteAprioriFile(Xnew,filename="AprioriFileProject03.txt")
runApriori("AprioriFileProject03.txt",minSup=1,minConf=10,maxRule=4)


