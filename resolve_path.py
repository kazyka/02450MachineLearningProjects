'''
Paths --
Toolbox path: ../02450 - Machine Learning/Toolbox/
Data path: ./insuranceCompany_Data/
    {'ticdata2000.txt', 'ticeval2000.txt', 'tictgts2000.txt'}
'''
import os
import sys

# Add path to course toolbox
abs_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.dirname(abs_path)

sys.path.append(par_path + '/02450 - Machine Learning/Toolbox/02450Toolbox_Python/Scripts')
sys.path.append(par_path + '/02450 - Machine Learning/Toolbox/02450Toolbox_Python/Tools')

# Check if data set exists
datafile = ['ticdata2000.txt', 'ticeval2000.txt', 'tictgts2000.txt']
for fname in datafile:
    if not os.path.isfile(abs_path + '/insuranceCompany_Data/' + fname):
        print(fname+' is missing!', file=sys.stderr)
sys.stderr.flush()
