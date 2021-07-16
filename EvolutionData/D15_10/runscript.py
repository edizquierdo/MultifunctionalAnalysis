import os
import sys

fr = int(sys.argv[1])
to = int(sys.argv[2])
currentpath = os.getcwd()
for k in range(fr,to):
    print(k)
    os.chdir(currentpath+'/'+str(k))
    os.system('../main')
    os.chdir('../')
