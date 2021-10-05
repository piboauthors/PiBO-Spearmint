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

    # return 0.7
    keys = ['a_x0', 'b_x1', "c_dropout", "d_beta_1", "e_beta_2", "f_activation"]
    keys = [unicode(key, "utf-8") for key in keys]
    param_list = [str(float(params[key][0])) for key in keys]
    print param_list
    cmd = ["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_unet.sh")]
    cmd = cmd + param_list
    p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = p.communicate()
    match = re.search(r"Start loss (.*?) end loss\n", stdout)
    print "stdout", stdout
    value = match.group(1)
    print("value printed ", value) 

    try:
        value = float(value)
    except:
        print("Something very weird happened")
        return 0.0

    if np.isnan(value):
        return 0.0
    else:
        return value
