import matplotlib as mpl
mpl.use("Agg")

from .utils import CollisionDataset, AutoencoderDataset, train, test, plot_curves
from . import nn_classes as nn