import tensorflow as tf
from tensorflow import keras


class Network:

    """
    Simple neural network for the estimation of the redshift for astronomical objects (quasar and galaxies)
    """

    def __init__(self):
        """
        Initialize network.
        """
        reg = tf.keras.regularizers.l2(l=1e-3)
        self.model = keras.Sequential([
            # scale brightness
            keras.layers.Dense(5, activation=tf.nn.leaky_relu),
            # calculate stuff
            keras.layers.Dense(15, activation=tf.nn.sigmoid, use_bias=True, kernel_regularizer=reg),
            keras.layers.Dense(30, activation=tf.nn.sigmoid, use_bias=True, kernel_regularizer=reg),
            keras.layers.Dense(15, activation=tf.nn.sigmoid, use_bias=True, kernel_regularizer=reg),

            keras.layers.Dense(1, activation=tf.nn.leaky_relu)
        ])

        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def train(self, input_data, output_data, val_in=None, val_out=None, epochs=30, **kwargs):
        """
        Train the network.
        :param input_data: np.array[n, 5] = List of input data to train
        :param output_data: np.array[n] = List of output data to train
        :param val_in: Optional[np.array[n]] = List of input data to validate
        :param val_out: Optional[np.array[n]] = List of output data to validate
        :param epochs: int = Number of interactions to train
        :return: history of training
        """
        return self.model.fit(input_data, output_data, validation_data=(val_in, val_out), epochs=epochs,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)], **kwargs)

    def predict(self, input_data):
        """
        Calculate the redshift from the given input data.
        :param input_data: np.array[5] = List of input data
        :return: float = Estimated redshift
        """
        return self.model.predict(input_data)
