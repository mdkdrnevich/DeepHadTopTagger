from sklearn.ensemble import GradientBoostingClassifier
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("training", help="File path to the training set")
parser.add_argument("validation", help="File path to the validation set")
parser.add_argument("-n", "--name", help="Name to help describe the output neural net and standardizer", default="")
args = parser.parse_args()

train = np.load(args.training)
val = np.load(args.validation)
train_x = train[:, 1:]
train_y = train[:, 0]
val_x = val[:, 1:]
val_y = val[:, 0]

params = dict(max_depth=8, learning_rate=0.1, n_estimators=1000, min_samples_leaf=0.045, subsample=0.5, min_samples_split=20)
bdt = GradientBoostingClassifier(**params).fit(train_x, train_y)

bdt.score(val_x, val_y)*100

with open("{}_bdt.pkl".format(args.name), 'wb') as f:
    pickle.dump(bdt, f)