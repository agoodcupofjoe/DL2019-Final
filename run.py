import subprocess

with open("scripts.txt", "r") as f:
    scripts = [line[:-2] for line in f.readlines()]
    for s in scripts:
        print("RUNNING THE FOLLOWING SCRIPT: {}".format(s))
        subprocess.call(s.split(), shell = False )
