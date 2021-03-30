import os
import sys

fr = int(sys.argv[1])
to = int(sys.argv[2])

for k in range(fr,to):
    print(k)
    os.system("mkdir "+str(k))
    os.system("cd "+str(k))
    os.system('../main')
    os.system("cd ../")
