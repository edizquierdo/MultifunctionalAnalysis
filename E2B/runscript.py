import os
import sys

fr = int(sys.argv[1])
to = int(sys.argv[2])
program = sys.argv[3]
currentpath = os.getcwd()
for k in range(fr,to):
    print(k)
    #os.system("mkdir "+str(k))
    os.chdir(currentpath+'/'+str(k))
    os.system('../'+program)
    os.chdir('../')
