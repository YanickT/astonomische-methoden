import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
from network import Network


def get_data(class_, n_train=1000, n_test=1000):
    """
    WHY THE HECK DO I GET AN ERROR IF I DO NOT SELECT specobj.ra???
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


# only galaxy
train_in, train_out, test_in, test_out = get_data("GALAXY", 3000, 1000)
net = Network()
history = net.train(train_in, train_out, 25)
mae = history.history["mean_absolute_error"]

plt.plot(list(range(len(mae))), mae)
plt.show()

preds = net.predict(test_in).flatten()
test = np.isclose(preds, test_out, rtol=0.2)
trues = np.count_nonzero(test)
false = test.shape[0] - trues
print(f"True: {trues}, False: {false}")


# only quasar
train_in, train_out, test_in, test_out = get_data("QSO", 3000, 1000)
net = Network()
history = net.train(train_in, train_out, 25)
mae = history.history["mean_absolute_error"]

plt.plot(list(range(len(mae))), mae)
plt.show()

preds = net.predict(test_in).flatten()
test = np.isclose(preds, test_out, rtol=0.2)
trues = np.count_nonzero(test)
false = test.shape[0] - trues
print(f"True: {trues}, False: {false}")
