import numpy as np
import torch
import matplotlib.pyplot as plt

from astroquery.sdss import SDSS
from network import Network
from torch.utils.data import TensorDataset, DataLoader


def get_data(class_, n_train=1000, n_test=1000, batchsize=128):
    """
    Load data for training and evaluation.
    :param class_: str = Type of astronomical objects
    :param n_train: int = Number of traingings data
    :return: np.array[n_train, 5], np.array[n_train], np.array[n_test, 5], np.array[n_test] =
    input-training, output-training, input-test, output-test
    """
    query = f"""SELECT top {n_train + n_test} specobj.z, specobj.ra, PhotoObj.u, PhotoObj.g, PhotoObj.r, PhotoObj.i, PhotoObj.z 
    FROM specObj JOIN PhotoObj ON specObj.bestObjID = PhotoObj.objid 
    WHERE class = '{class_}' AND specobj.z > -0.1 AND NOT bestObjID = 0 AND zWarning = 0 ORDER BY ra"""
    data = np.array([list(tup) for tup in np.array(SDSS.query_sql(query)).tolist()])
    outputs = data[:, 0]
    inputs = data[:, 2:]

    train_out = torch.tensor(outputs[:n_train].reshape((n_train, 1)), dtype=torch.float)
    train_in = torch.tensor(inputs[:n_train, :], dtype=torch.float)
    test_out = torch.tensor(outputs[n_train:].reshape((n_test, 1)), dtype=torch.float)
    test_in = torch.tensor(inputs[n_train:, :], dtype=torch.float)

    return DataLoader(TensorDataset(train_in, train_out), batch_size=batchsize), \
           DataLoader(TensorDataset(test_in, test_out), batch_size=batchsize)


def redshift_error_plot(preds, test_out):
    """
    Create plot of the uncertainty over the redshift.
    :param net: Network = Network to use
    :param test_in: np.array[5, n] = List of input data to test
    :param test_out: np.array[n] = List of output data to test
    :return: void
    """
    preds, test_out = np.array(preds), np.array(test_out)
    diff = np.abs(preds - test_out)

    plt.plot(test_out, diff / np.array(test_out), "x")
    plt.xlabel("redshift")
    plt.ylabel("rel error")
    plt.show()


# only galaxy
if True:
    train_set, test_set = get_data("GALAXY", 10000, 3000, 100)
    net = Network()
    val_loss = net.train(60, train_set, test_set)

    xs = []
    ys = []
    for input_, output in test_set:
        xs += output.numpy().flatten().tolist()
        ys += net.predict(input_).numpy().flatten().tolist()

    plt.plot(xs, ys, "x")
    plt.xlabel("redshift")
    plt.ylabel("predicted redshift")
    plt.show()

    # plot loss of test data
    plt.plot(list(range(len(val_loss))), val_loss)
    plt.xlabel("epochs")
    plt.ylabel("validation loss")
    plt.show()

    # plot loss of test data over redshift
    redshift_error_plot(ys, xs)


# only quasar
if True:
    """
    Use with regularization in Network
    """
    train_set, test_set = get_data("QSO", 10000, 3000, 100)
    net = Network()
    val_loss = net.train(60, train_set, test_set)

    xs = []
    ys = []
    for input_, output in test_set:
        xs += output.numpy().flatten().tolist()
        ys += net.predict(input_).numpy().flatten().tolist()

    plt.plot(xs, ys, "x")
    plt.xlabel("redshift")
    plt.ylabel("predicted redshift")
    plt.show()

    # plot loss of test data
    plt.plot(list(range(len(val_loss))), val_loss)
    plt.xlabel("epochs")
    plt.ylabel("validation loss")
    plt.show()

    # plot loss of test data over redshift
    redshift_error_plot(ys, xs)

