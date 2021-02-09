import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb


def squared_relative_loss(y_actual,y_pred):
    loss = kb.square(y_actual-y_pred)
    loss /= y_actual
    return loss


class Network:

    """
    Simple neural network for the identification of letters in images.
    """

    def __init__(self):
        """
        Initialize network.
        """
        self.model = keras.Sequential([
            # scale brightness
            keras.layers.Dense(5, activation=tf.nn.leaky_relu),
            # calculate stuff
            keras.layers.Dense(15, activation=tf.nn.sigmoid),
            keras.layers.Dense(20, activation=tf.nn.sigmoid),
            keras.layers.Dense(15, activation=tf.nn.sigmoid),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self.model.compile(optimizer='adam', loss=squared_relative_loss,
                           metrics=[tf.keras.metrics.MeanAbsoluteError(),
                                    tf.keras.metrics.MeanAbsolutePercentageError()])

    def train(self, input_data, output_data, epochs=5):
        """
        Train the network.
        :param input_data: np.array[n, 5] = List of input data
        :param output_data: np.array[n] = List of output data
        :param epochs: int = Number of interactions to train
        :return: history of training
        """
        return self.model.fit(input_data, output_data, epochs=epochs)

    def predict(self, input_data):
        """
        Calculate the redshift from the given input data.
        :param input_data: np.array[5] = List of input data
        :return: float = Estimated redshift
        """
        return self.model.predict(input_data)