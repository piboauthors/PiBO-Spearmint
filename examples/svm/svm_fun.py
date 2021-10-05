import numpy as np
import sys
import math
import csv
import os
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))),"libs/emukit"))
from emukit.examples.profet.meta_benchmarks import meta_svm

def svm_function(x0, x1):
    path_to_files = join(dirname(dirname(abspath(__file__))),"libs/profet_data")
    function_family = "svm"
    function_id = 42
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    fcn, parameter_space = meta_svm(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    X = np.array([[x0, x1]])
    result = fcn(X)
    try:
        y, c = result # function value, cost for all functions except forrester
    except ValueError:
        y = result
    return float(y[0,0])


if __name__ == "__main__":
    x0 = float(sys.argv[1])
    x1 = float(sys.argv[2])
    value = svm_function(x0, x1)
    print(value)

