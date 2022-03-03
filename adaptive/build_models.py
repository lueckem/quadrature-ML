import tensorflow as tf


def build_value_model(dim_state, dim_action, filename=None, lr=0.001, memory=0):
    """
    Build the neural network model for predicting the step size.

    Parameters
    ----------
    dim_state : int
    dim_action : int
    filename : str, optional
        load weights that are saved in filename
    lr : float, optional
        learning rate
    memory : int, optional
        how many iterations in the past are saved in the state

    Returns
    -------
    tf.keras.Model
    """

    dim_state = dim_state * (memory + 1)
    n_hidden = 5 * dim_state

    inputs = tf.keras.Input(shape=(dim_state,), name='state')
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(inputs)
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    outputs = tf.keras.layers.Dense(dim_action)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    if filename is not None:
        model.load_weights(filename)

    return model


def build_value_modelODE(dim_state, dim_action, filename=None, lr=0.001, memory=0):
    """
    Build the neural network model for predicting the step size.

    Parameters
    ----------
    dim_state : int
    dim_action : int
    filename : str, optional
        load weights that are saved in filename
    lr : float, optional
        learning rate
    memory : int, optional
        how many iterations in the past are saved in the state

    Returns
    -------
    tf.keras.Model
    """

    dim_state = dim_state * (memory + 1)
    n_hidden = 5 * dim_state

    inputs = tf.keras.Input(shape=(dim_state,), name='state')
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(inputs)  # !regularizer!
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    x = tf.keras.layers.Dense(n_hidden, activation='relu')(x)
    outputs = tf.keras.layers.Dense(dim_action)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')  # !huber loss!

    if filename is not None:
        model.load_weights(filename)

    return model


def build_estimator_model(dim_state, filename=None, lr=0.001):
    """
    Build the neural network model for estimating the error.

    Parameters
    ----------
    dim_state : int
    filename : str, optional
        load weights that are saved in filename
    lr : float, optional
        learning rate

    Returns
    -------
    tf.keras.Model
    """

    n_hidden1 = 20
    n_hidden2 = 20

    inputs = tf.keras.Input(shape=(dim_state,), name='state')
    x = tf.keras.layers.Dense(n_hidden1, activation='relu')(inputs)
    x = tf.keras.layers.Dense(n_hidden2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if filename is not None:
        model.load_weights(filename)

    return model
