import numpy as np
import sys
import math
import csv
import os
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))),"libs/emukit"))
from emukit.examples.profet.meta_benchmarks import meta_fcnet

def fcnet_function(x0, x1, x2, x3, x4, x5):
    path_to_files = join(dirname(dirname(abspath(__file__))),"libs/profet_data")
    function_family = "fcnet"
    function_id = 42
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    fcn, parameter_space = meta_fcnet(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    X = np.array([[x0, x1, x2, x3, x4, x5]])
    result = fcn(X)
    try:
        y, c = result # function value, cost for all functions except forrester
    except ValueError:
        y = result
    return float(y[0,0])


if __name__ == "__main__":
    x0 = float(sys.argv[1])
    x1 = float(sys.argv[2])
    x2 = float(sys.argv[3])
    x3 = float(sys.argv[4])
    x4 = float(sys.argv[5])
    x5 = float(sys.argv[6])
    value = fcnet_function(x0, x1, x2, x3, x4, x5)
    print(value)

