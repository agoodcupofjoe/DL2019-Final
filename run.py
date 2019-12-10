import subprocess
import glob
from operator import itemgetter 

with open("scripts.txt", "r") as f:
    scripts = [line[:-1] for line in f.readlines()]
    for s in scripts:
        print("RUNNING THE FOLLOWING SCRIPT: {}".format(s))
        subprocess.call(s.split(), shell = False )

logs = glob.glob("log/*/*/log.txt")

all = []
for log in logs:
    with open(log, "r") as l:
        temp = []
        for line in l.readlines():
            temp.append(line[:-1].split(": ")[1])
        all.append(temp)
all = sorted(all, key = itemgetter(1))
all = sorted(all, key = itemgetter(0))

with open("logs.csv", "w+") as f:
    header = ['Model', 'Loss', 'Global Accuracy', 'Sn', 'PPV', 'F1']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all)
