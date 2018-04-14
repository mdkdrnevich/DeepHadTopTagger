from __future__ import print_function, division
import pandas as pd
import numpy as np
import torch as th


def score(model, dataset, cut=0.5):
    X, y = Variable(dataset[:]['input']).float(), dataset[:]['target'].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    predicted = (out >= cut).type(th.ByteTensor)
    return (predicted == y).sum()/out.size()[0]


def find_cut(model, dataset, n_steps=100, benchmark="f1"):
    X, y = Variable(dataset[:]['input']).float(), dataset[:]['target'].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    best_cut = 0
    best_score = -1
    for i in range(n_steps+1):
        cut = i*((out.max() - out.min())/n_steps) + out.min()
        if benchmark == "f1":
            score = f1_score(y.numpy(), (out >= cut).type(th.ByteTensor).numpy())
        elif benchmark == "acc":
            score = ((out >= cut).type(th.ByteTensor) == y).sum()
        if score > best_score:
            best_score = score
            best_cut = cut
    return best_cut