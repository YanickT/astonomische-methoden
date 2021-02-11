import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb


def squared_relative_loss(y_actual,y_pred):
    """
    Calculates the relative squared error. This should cause a constant relative uncertainty in the predictions.
    :param y_actual: np.array[1] = Correct output data
    :param y_pred: np.array[1] = Predicted output data
    :return: np.array[1] = loss
    """
    loss = kb.square((y_actual-y_pred) / y_actual)
    return loss


class Network:

    """
    Simple neural network for the estimation of the redshift for astronomical objects (quasar and galaxies)
    """

    def __init__(self):
        """
        Initialize network.
        """
        self.model = keras.Sequential([
            # scale brightness
            keras.layers.Dense(5, activation=tf.nn.leaky_relu),
            # calculate stuff
            keras.layers.Dense(20, activation=tf.nn.sigmoid),
            keras.layers.Dense(60, activation=tf.nn.sigmoid),
            keras.layers.Dense(20, activation=tf.nn.sigmoid),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
            # , kernel_regularizer=keras.regularizers.l2(l=0.1)
        ])

        self.model.compile(optimizer='adam', loss=squared_relative_loss,
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def train(self, input_data, output_data, val_in=None, val_out=None, epochs=5, **kwargs):
        """
        Train the network.
        :param input_data: np.array[n, 5] = List of input data to train
        :param output_data: np.array[n] = List of output data to train
        :param val_in: Optional[np.array[n]] = List of input data to validate
        :param val_out: Optional[np.array[n]] = List of output data to validate
        :param epochs: int = Number of interactions to train
        :return: history of training
        """
        if val_in is not None and val_out is not None:
            return self.model.fit(input_data, output_data, validation_data=(val_in, val_out), epochs=epochs, **kwargs)
        else:
            return self.model.fit(input_data, output_data, epochs=epochs, **kwargs)

    def predict(self, input_data):
        """
        Calculate the redshift from the given input data.
        :param input_data: np.array[5] = List of input data
        :return: float = Estimated redshift
        """
        return self.model.predict(input_data)
