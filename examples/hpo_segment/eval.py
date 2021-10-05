import sys
import os
from os.path import join, dirname, abspath
import numpy as np
import ConfigSpace
import warnings
warnings.filterwarnings('ignore')
from ConfigSpace import Configuration

sys.path.append(join(dirname(dirname(abspath(__file__))),"libs/HPOBench"))
from hpobench.container.benchmarks.ml.tabular_benchmark import TabularBenchmark as Benchmark


def format_arguments(config_space, args):
    order = ['alpha', 'batch_size', 'depth', 'learning_rate_init', 'width']
    args_dict = {o: int(a) for o, a in zip(order, args)}
    formatted_hparams = {}
    for param in args_dict.keys():
        formatted_hparams[param] = config_space.get_hyperparameter(param).sequence[args_dict[param]]
    return formatted_hparams

def eval_function(args):
    
    b = Benchmark(task_id=146822, rng=1, model='nn')
    config_space = b.get_configuration_space(seed=1)
    args_dict = format_arguments(config_space, args)
    config = Configuration(b.get_configuration_space(seed=1), args_dict)

    result_dict = b.objective_function(configuration=config, rng=1)
    return result_dict['function_value']

if __name__ == "__main__":
    value = eval_function(sys.argv[1:])
    print('FunctionResult', value, 'FunctionResult')

