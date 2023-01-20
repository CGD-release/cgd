import numpy as np

import argparse
import os
import torch
import random
from CGD_BDP import CGD
import pickle
from bayesian_privacy_accountant import BayesianPrivacyAccountant
from constants import *

if __name__ == '__main__':

    with open("./pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]

    num_iter = 2000  # number of iterations of gradient descent default: 2000
    Pv = 4
    Ph = 4
    n_H = 256  # number of neurons in the hidden layer
    delta = 1e-1 #"random" #1e-1
    sampling_prob = 0.1
    max_grad_norm = 1
    sigma = 1e-2
    batch_num = 10
    cgd=CGD(num_iter,
            train_imgs,
            train_labels,
            train_labels_one_hot,
            test_imgs,
            test_labels,
            test_labels_one_hot,
            Ph,
            Pv,
            n_H,
            delta,
            'mnist',
            sampling_prob,
            max_grad_norm,
            sigma,
            batch_num)

    total_steps = num_iter * batch_num
    bayes_accountant = BayesianPrivacyAccountant(powers=[2, 4, 8, 16, 32], total_steps=total_steps)

    cgd.train(bayes_accountant)
    #cgd.train()

