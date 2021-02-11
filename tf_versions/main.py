import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
from network import Network


def get_data(class_, n_train=1000, n_test=1000):
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

    train_out = outputs[:n_train]
    train_ins = inputs[:n_train, :]
    test_out = outputs[n_train:]
    test_ins = inputs[n_train:, :]
    return train_ins, train_out, test_ins, test_out


def redshift_error_plot(net, test_in, test_out):
    """
    Create plot of the uncertainty over the redshift.
    :param net: Network = Network to use
    :param test_in: np.array[5, n] = List of input data to test
    :param test_out: np.array[n] = List of output data to test
    :return: void
    """
    preds = net.predict(test_in).flatten()
    diff = np.abs(preds - test_out)
    plt.plot(test_out, np.array(diff) / np.array(test_out), ".")
    plt.ticklabel_format(style='plain')
    plt.xlabel("redshift")
    plt.ylabel("rel error")
    plt.show()


# only galaxy
if False:
    train_in, train_out, test_in, test_out = get_data("GALAXY", 10000, 3000)
    net = Network()
    history = net.train(train_in, train_out, val_in=test_in, val_out=test_out, epochs=60, verbose=0)
    net.model.save("galaxies.h5")

    preds = net.predict(test_in)
    plt.plot(test_out, preds.flatten(), "x")
    plt.xlabel("redshift")
    plt.ylabel("predicted redshift")
    plt.show()

    # plot loss of test data
    val_loss = history.history["val_loss"]
    plt.plot(list(range(len(val_loss))), val_loss)
    plt.xlabel("epochs")
    plt.ylabel("validation loss")
    plt.show()

    # plot loss of test data over redshift
    redshift_error_plot(net, test_in, test_out)


# only quasar
if True:
    """
    Use with regularization in Network
    """
    train_in, train_out, test_in, test_out = get_data("QSO", 10000, 3000)
    net = Network()
    history = net.train(train_in, train_out, val_in=test_in, val_out=test_out, epochs=60, verbose=0)
    net.model.save("quasar.h5")

    preds = net.predict(test_in)
    plt.plot(test_out, preds.flatten(), "x")
    plt.xlabel("redshift")
    plt.ylabel("predicted redshift")
    plt.show()

    # plot loss of test data
    val_loss = history.history["val_loss"]
    plt.plot(list(range(len(val_loss))), val_loss)
    plt.xlabel("epochs")
    plt.ylabel("validation loss")
    plt.show()

    # plot loss of test data over redshift
    redshift_error_plot(net, test_in, test_out)

