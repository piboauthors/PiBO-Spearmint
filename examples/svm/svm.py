import numpy as np
import math
import csv
import os
import subprocess
from subprocess import PIPE, STDOUT
import re


# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params

    keys = ['x0', 'x1']
    keys = [unicode(key, "utf-8") for key in keys]
    param_list = [str(float(params[key][0])) for key in keys]
    cmd = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm_fun.py")]
    cmd = cmd + param_list
    p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = p.communicate()
    print(stdout)
    stdout = stdout.split("\n")[0]
    value = float(stdout)

    return value
