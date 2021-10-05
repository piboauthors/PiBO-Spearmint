import ConfigSpace

# import horovod.tensorflow as hvd

from unet_medical.model.unet import Unet
from unet_medical.runtime.run import train, evaluate
from unet_medical.runtime.setup import get_logger, set_flags, prepare_model_dir
from unet_medical.runtime.arguments import PARSER, parse_args
from unet_medical.data_loading.data_loader import Dataset

import tensorflow as tf

from munch import Munch

import random
import uuid
import os


def evaluate_unet(config):
    # Adapted from script with license:
    # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
    # Licensed under the Apache License, Version 2.0 (the "License")
    # hvd.init()
    x_dir = f"/tmp/models/{uuid.uuid4()}"
    params = Munch({
        'exec_mode': "train_and_evaluate",
        'model_dir': x_dir,
        'data_dir': f"{os.environ.get('HOME')}/spearmint_priors/data_unet",
        'log_dir': x_dir + "/log.json",
        'batch_size': 8,
        'learning_rate': 10 ** config["learning_rate"],
        'fold': 0,
        'max_steps': 6400,
        'weight_decay': 10 ** config["weight_decay"],
        'log_every': 400,
        'evaluate_every': 0,
        'warmup_steps': 0,
        'augment': True,
        'benchmark': False,
        'seed': random.seed(),
        'use_amp': True,
        'use_trt': False,
        'use_xla': True,
        'resume_training': False,
        "dropout": config["dropout"],
        "beta_1": 1 - 10 ** -config["beta_1"],
        "beta_2": 1 - 10 ** -config["beta_2"],
        "activation": config["activation"]
    })

    set_flags(params)
    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    logger = get_logger(params)

    model = Unet(params)
    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.fold,
                      augment=params.augment,
                      gpu_id=0,#hvd.rank(),
                      num_gpus=1,#hvd.size(),
                      seed=params.seed)
    train(params, model, dataset, logger)
    dice_score = evaluate(params, model, dataset, logger)  # if hvd.rank() == 0:
    return -dice_score


if __name__ == '__main__':
    import sys

    original_stdout = sys.stdout
    sys.stdout = sys.stderr
   
    lr = float(sys.argv[1])
    wd = float(sys.argv[2])
    dropout = float(sys.argv[3])
    beta_1 = float(sys.argv[4])
    beta_2 = float(sys.argv[5])
    if int(float(sys.argv[6])) == 0:
        activation = tf.nn.relu
    elif int(float(sys.argv[6])) == 1:
        activation = tf.nn.leaky_relu
    elif int(float(sys.argv[6])) == 2:
        activation = tf.nn.selu
    else:
        raise ValueError
    loss = evaluate_unet(dict(learning_rate=lr, weight_decay=wd, dropout=dropout, beta_1=beta_1, beta_2=beta_2, activation=activation))

    sys.stdout = original_stdout
    print("Start loss ", loss, " end loss\n")

