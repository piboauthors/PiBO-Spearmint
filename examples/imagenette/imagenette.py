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
    keys = [
        "a_lr",
        "b_sqmom",
        "c_mom",
        "d_eps",
        "e_mixup",
        "f_pool"
    ]
    keys = [unicode(key, "utf-8") for key in keys]
    p = [params[key][0] for key in keys]
    params_list = [
        10 ** float(p[0]),
        1 - 10 ** -float(p[1]),
        float(p[2]),
        10 ** float(p[3]),
        float(p[4]),
        "MaxPool" if int(p[5]) == 1 else "AvgPool",
    ]
    params_list = [str(x) for x in params_list]
    print params_list
    cmd = ["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_imagenette.sh")]
    cmd = cmd + params_list
    p = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    print "stdout"
    print
    print stdout
    print
    print
    print "stderr"
    print
    print stderr
    print
    match = re.search(r"79        \d\.\d\d\d\d\d\d    \d\.\d\d\d\d\d\d    (\d\.\d\d\d\d\d\d)  \d\.\d\d\d\d\d\d", stdout)
    try:
        value = match.group(1)
    except:
        print("match.group failed")
        return 0.0
    print("value printed ", value) 
    try:
        value = -float(value)
    except:
        print("Something very weird happened")
        return 0.0

    if np.isnan(value):
        return 0.0
    else:
        return value
