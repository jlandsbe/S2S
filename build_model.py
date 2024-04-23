"""Data processing.

Functions
---------

"""

import numpy as np
import tensorflow as tf
import metrics
import gc
from keras.utils.generic_utils import get_custom_objects
import tensorflow_probability as tfp

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"


class SampleSoftmax(tf.keras.Model):
    def __init__(self, tau=.01):
        super(SampleSoftmax, self).__init__()
        self.tau = tau
        self.dist = tfp.distributions.Gumbel(loc=0, scale=1)

    def call(self, x, training=None):
        # Generate Gumbel noise
        #noise = .1*self.dist.sample(tf.shape(x))
        #noisy_x = x + noise
        noisy_x = x

        # Apply softmax temperature
        noisy_x = noisy_x / self.tau
        output = tf.nn.softmax(noisy_x)
        # Compute softmax
        

        return output

class CustomRidgeRegularization(tf.keras.regularizers.Regularizer):
    def __init__(self, lambda_param):
        self.lambda_param = lambda_param

    def __call__(self, x):
        return self.lambda_param / tf.reduce_sum(tf.square(x))

    def get_config(self):
        return {'lambda_param': float(self.lambda_param)}
def create_blur_filter(filter_size=4):
    """Create a 2D blur filter (Gaussian blur) with the specified size."""
    blur_filter = np.ones((filter_size, filter_size), dtype=np.float32)
    blur_filter /= filter_size ** 2
    return blur_filter.reshape(filter_size, filter_size, 1, 1)
def smooth_tensor(input_tensor):
    blur_filter = create_blur_filter()
    
    # Expand filter to match input tensor channels
    num_channels = input_tensor.shape[-1]
    blur_filter = np.tile(blur_filter, (1, 1, num_channels, 1))
    
    # Convert blur filter to TensorFlow constant
    blur_filter_tf = tf.constant(blur_filter, dtype=tf.float32)
    
    # Perform depthwise convolution for each channel
    output = tf.nn.depthwise_conv2d(
        input=input_tensor,
        filter=blur_filter_tf,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    return output



def pairwise_distances(matrix, map_dim, n_branches, strength=50):
        
    weights_mean = tf.cast(tf.math.reduce_mean(matrix, axis=-2, keepdims=True), dtype=tf.float32)
    weights_weighting = tf.cast(tf.math.reciprocal_no_nan(weights_mean), dtype=tf.float32)
    matrix = tf.math.multiply(weights_weighting, tf.cast(matrix, dtype=tf.float32)) 
    matrix = tf.keras.layers.Reshape((map_dim[0],map_dim[1],n_branches))(matrix)
    matrix = smooth_tensor(matrix)
    matrix = tf.keras.layers.Reshape((map_dim[0]*map_dim[1] ,n_branches))(matrix)
    matrix = tf.math.reduce_mean(matrix, axis=0)

    matrix = tf.transpose(matrix)
    expanded_tensor_data = tf.expand_dims(matrix, axis=1)  # Shape: (num_samples, 1, num_features)
    subtracted = expanded_tensor_data - tf.transpose(expanded_tensor_data, perm=[1, 0, 2])  # Pairwise differences
    distances = tf.norm(subtracted, axis=-1)
    distances = tf.nn.relu(distances)
    rows, cols = tf.shape(distances)[0], tf.shape(distances)[1]
    mask = tf.eye(rows, cols, dtype=tf.bool)
    
    # Invert mask to get non-diagonal elements
    non_diagonal_mask = tf.logical_not(mask)
    
    non_diagonal_mask = tf.reshape(non_diagonal_mask, [rows, cols])
    
    non_diagonal_values = tf.boolean_mask(distances, non_diagonal_mask)
    
    min_value = tf.reduce_min(non_diagonal_values)
    

    output = strength/(min_value+.0001)
    
    return output


def JBLsoftmax(x, axis=-1, training=False):
    #if training:
    if 1:
        x_norm = (x - tf.reduce_max(x, axis=axis, keepdims=True))
        numerator = tf.math.pow(20.0,x_norm/.01)
        denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
        output = numerator / (denominator)
    else:
        x_norm = (x - tf.reduce_max(x, axis=axis, keepdims=True))
        numerator = tf.math.pow(10.0,x_norm)
        denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
        output = numerator / (denominator + .0001)
        mn = tf.reduce_min(x, axis=axis, keepdims=True)
        xplus = tf.math.subtract(x,mn)
        mx = tf.reduce_max(xplus, axis=axis, keepdims=True)
        output = tf.math.floor(tf.math.divide(xplus,mx))
    return output

get_custom_objects().update({'JBLcustom': tf.keras.layers.Activation(JBLsoftmax)})

def parse_model(x_input, mask_model, dissimilarity_model, prediction_model):
    weighted_soi, weighted_analog, weights, gates, branches = mask_model(x_input)
    dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
    prediction = prediction_model([dissimilarities])
    __ = gc.collect()  # to fix memory leak

    return weights.numpy(), dissimilarities.numpy(), prediction.numpy(), gates.numpy(), branches.numpy()

def std_dev_regularizer(x): 
    return 1e11*(1/(tf.keras.backend.std(x)+.1))

def non_zero_regularizer(x):
    # Compute the regularization term based on the number of non-zero elements
    regularization_term = 1e-5 * tf.keras.backend.sum(tf.keras.backend.not_equal(x, 0))
    return regularization_term

def make_weights_model(x_train, mask_model_act, mask_l1, mask_l2, mask_initial_value, normalize_weights_bool, state_bool, weight_nodes=(), weight_act = "relu", cnn = 0, n_branches = 1, rng_seed=1):

    map_dim = x_train[0].shape[1:]

    soi_input_layer = tf.keras.layers.Input(
        shape=map_dim, name='soi_input'
        )  # shape is lat x lon x channels
    analog_input_layer = tf.keras.layers.Input(
        shape=map_dim,
        name='analogs_input'
        )  # shape is ensemble members x lat x lon x channels

    # Flatten Layer
    if cnn:
        num_filters = 32  
        x = soi_input_layer
        x2 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

        x2 = tf.keras.layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(x2)
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

        x2 = tf.keras.layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(x2)
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

        x2 = tf.keras.layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(x2)
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

        x2 = tf.keras.layers.Flatten()(x2)

        x2 = tf.keras.layers.Dense(10, activation='relu',
                                kernel_initializer=tf.keras.initializers.Ones(),
                                bias_initializer=tf.keras.initializers.Ones())(x2)
        x2 = tf.keras.layers.Dense(10, activation='relu',
                        kernel_initializer=tf.keras.initializers.Ones(),
                        bias_initializer=tf.keras.initializers.Ones())(x2)
        x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(10, activation='relu',
                                kernel_initializer=tf.keras.initializers.Ones(),
                                bias_initializer=tf.keras.initializers.Ones())(x)
        x = tf.keras.layers.Dense(10, activation='relu',
                        kernel_initializer=tf.keras.initializers.Ones(),
                        bias_initializer=tf.keras.initializers.Ones())(x)
        
        # x = tf.keras.layers.Dense(1, activation='relu',
        #                         kernel_initializer=tf.keras.initializers.Ones(),
        #                         bias_initializer=tf.keras.initializers.Ones())(x)
     
    else:
        x = tf.keras.layers.Flatten()(soi_input_layer)
        for nodes in weight_nodes:
            x = tf.keras.layers.Dense(
            np.product(map_dim), activation='relu',
            #kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+5),
            #bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+6),
            kernel_initializer=tf.keras.initializers.Ones(),
            bias_initializer=tf.keras.initializers.Ones(),
            )(x)
    
    if 1:
        stem = tf.keras.layers.Dense(np.product(map_dim), activation='linear',
                kernel_initializer=tf.keras.initializers.Ones(), use_bias=False)(x)
        stem.trainable = False
        class BranchLayer(tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                super(BranchLayer, self).__init__(*args, **kwargs)
                
            def build(self, input_shape):
                self.bias = self.add_weight('bias',
                                            shape=(np.product(map_dim)),
                                            initializer=mask_initial_value,
                                            trainable=True,
                                            constraint=tf.keras.constraints.NonNeg(),)
            def call(self, x):
                return tf.zeros_like(x) + self.bias
            
        #make branches
        branches = []
        #make gate
        gate = tf.keras.layers.Dense(n_branches, activation="linear",
                        kernel_initializer=tf.keras.initializers.RandomNormal(),
                        use_bias = False, activity_regularizer=tf.keras.regularizers.l1(1))(x2)
        gate = (tf.keras.layers.Lambda(JBLsoftmax))(gate)
        c = .10
        gate = c * tf.keras.layers.GaussianNoise(1)(gate)
        sample_softmax = SampleSoftmax(.00001)
        norm_gate = sample_softmax(gate)



        for b in range(n_branches):
            weights = BranchLayer()
            wl = weights(stem)
            wl = tf.keras.layers.Activation(mask_model_act)(wl)
            wl = tf.keras.layers.ActivityRegularization(l1=mask_l1, l2=mask_l2)(wl)
            branches.append(wl)
        branches = tf.stack(branches,axis=-1)
        loss_pen = pairwise_distances(branches, map_dim, n_branches)
        bout = branches
        branches_weighted = tf.keras.layers.Multiply()([tf.cast(branches,tf.float32), tf.cast(norm_gate,tf.float32)])
# Sum the weighted branches to keep only one branch
        weights_layer = tf.math.reduce_sum(branches_weighted, axis=-1)
        #need to 0 out all but 1 branches, and let 1 branch go through to next layer
        


        

    elif not state_bool:
    #if 1:
        # Bias-only layer (e.g. inputs don't affect mask)
        class WeightsLayer(tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                super(WeightsLayer, self).__init__(*args, **kwargs)
                
            def build(self, input_shape):
                self.bias = self.add_weight('bias',
                                            shape=(input_shape[1:]),
                                            initializer=mask_initial_value,
                                            trainable=True,
                                            constraint=tf.keras.constraints.NonNeg(),)
            def call(self, x):
                return tf.zeros_like(x) + self.bias
            
        weights = WeightsLayer()
        weights_layer = weights(x)
        weights_layer = tf.keras.layers.Activation(mask_model_act)(weights_layer)
        weights_layer = tf.keras.layers.ActivityRegularization(l1=mask_l1, l2=mask_l2)(weights_layer)
    # else:
    #     #get_custom_objects().update({'relu_plus_c': Activation(relu_plus_c)})
    #     weights_layer = tf.keras.layers.Dense(np.product(map_dim), activation="relu",
    #     kernel_initializer=tf.keras.initializers.Constant(value=mask_initial_value),
    #     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0., l2=0.),
    #     activity_regularizer=tf.keras.regularizers.l1_l2(l1=mask_l1, l2=mask_l2),
    #     use_bias=False,
    #     )(x)
        #x = tf.keras.layers.ActivityRegularization(l1=mask_l1, l2=mask_l2)(x)
    if normalize_weights_bool:
        # do not need float64 if the weights values are not restricted
        weights_mean = tf.cast(tf.math.reduce_mean(weights_layer, axis=-1, keepdims=True), dtype=tf.float32)
        weights_weighting = tf.cast(tf.math.reciprocal_no_nan(weights_mean), dtype=tf.float32)
        weights_layer = tf.math.multiply(weights_weighting, tf.cast(weights_layer, dtype=tf.float32)) 

    # multiply weights layer by soi and analog inputs
    weights_layer = tf.keras.layers.Reshape(map_dim)(weights_layer)
    if not state_bool:
        weights_layer = tf.keras.layers.Layer()(weights_layer, name='weights_layer')
    weighted_soi = tf.keras.layers.multiply([weights_layer, soi_input_layer], name='weighted_soi')
    weighted_analog = tf.keras.layers.multiply([weights_layer, analog_input_layer], name='weighted_analogs')
    bout = tf.keras.layers.Reshape((map_dim[0],map_dim[1],n_branches))(bout)
    # Define Model
    mask_model = tf.keras.Model(
        inputs=[soi_input_layer, analog_input_layer],
        outputs=[weighted_soi, weighted_analog, weights_layer, norm_gate, bout],
        name="mask_model",
        )

    return mask_model, soi_input_layer, analog_input_layer, loss_pen


#  Second, creating the model that calculates the dissimilarity between the weighted maps
def make_dissimilarity_model(x_train):
    map_dim = x_train[0].shape[1:] # shape 
    weighted_soi_input_layer = tf.keras.layers.Input(shape=map_dim)
    weighted_analog_input_layer = tf.keras.layers.Input(shape=map_dim)
    # Calculate the MSE between the weighted SOI and the weighted analogs
    weighted_soi_flat = tf.keras.layers.Flatten()(weighted_soi_input_layer)
    weighted_analog_flat = tf.keras.layers.Flatten()(weighted_analog_input_layer) #maybe we don't want to flatten it out

    dissimilarity = metrics.mse(weighted_analog_flat, weighted_soi_flat)
    #dissimilarity = metrics.mse(weighted_analog_input_layer, weighted_soi_input_layer)
    dissimilarity = tf.keras.layers.Reshape((1,))(dissimilarity)

    dissimilarity_model = tf.keras.Model(
        inputs=[weighted_soi_input_layer, weighted_analog_input_layer],
        outputs=[dissimilarity],
        name="dissimilarity_model"
        )

    return dissimilarity_model


# Third, creating the model that uses the dissimilarity score to predict the dissimilarity of the maps 5 years later
def make_prediction_model(prediction_model_nodes, prediction_model_act, rng_seed, output_activation="linear"):
    dissimilarity_input_layer = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Layer()(dissimilarity_input_layer)
    #prediction = x
    # Add all the Dense Layers
    # for nodes in prediction_model_nodes:
    #     x = tf.keras.layers.Dense(
    #         nodes, activation=prediction_model_act,
    #         kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+4),
    #         bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+1),
    #         # kernel_initializer=tf.keras.initializers.Ones(),
    #         # bias_initializer=tf.keras.initializers.Ones(),
    #         )(x)

    # # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation=output_activation,
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+2),
        )(x)


    prediction_model = tf.keras.Model(
        inputs=[dissimilarity_input_layer],
        outputs=[prediction],
        name='prediction_model'
        )

    return prediction_model


# Combining all three models
def build_interp_model(settings, x_train):

    if settings["output_type"] == "classification":
        output_activation = "sigmoid"
    elif settings["output_type"] == "regression":
        output_activation = "linear"
    else:
        raise NotImplementedError("no such output activation")

    mask_model, soi_input, analog_input, loss_pen = make_weights_model(x_train,
                                                             mask_model_act=settings["mask_model_act"],
                                                             mask_l1=settings["mask_l1"],
                                                             mask_l2=settings["mask_l2"],
                                                             mask_initial_value=settings["mask_initial_value"],
                                                             normalize_weights_bool=settings["normalize_weights_bool"],
                                                             state_bool = settings["state_masks"],
                                                             weight_nodes=settings["weight_nodes"], 
                                                             weight_act = settings["weight_act"],
                                                             cnn = settings["cnn"], 
                                                             n_branches = settings["n_branches"]
                                                             )
    dissimilarity_model = make_dissimilarity_model(x_train)
    prediction_model = make_prediction_model(prediction_model_nodes=settings["prediction_model_nodes"],
                                             prediction_model_act=settings["prediction_model_act"],
                                             rng_seed=settings["rng_seed"],
                                             output_activation=output_activation,
                                             )

    weighted_soi, weighted_analog, weights, gates, branches = mask_model([soi_input, analog_input])
    dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
    prediction = prediction_model([dissimilarities])

    full_model = tf.keras.Model(
        inputs=[soi_input, analog_input],
        outputs=[prediction],
        name='full_model'
        )
    full_model.add_loss(loss_pen)

    return full_model, mask_model, dissimilarity_model, prediction_model


def build_ann_analog_model(settings, x_train):

    tf.keras.utils.set_random_seed(settings["rng_seed"])

    map_dim = x_train[0].shape[1:]
    soi_input = tf.keras.layers.Input(shape=map_dim, name='soi_input')
    analog_input = tf.keras.layers.Input(shape=map_dim, name='analog_input')

    # Build Model
    x_input = tf.keras.layers.Concatenate(axis=-1)([soi_input, analog_input])
    x = tf.keras.layers.Flatten()(x_input)

    # Add all the Dense Layers
    for layer, nodes in enumerate(settings["ann_analog_model_nodes"]):

        if layer == 0:
            input_l2 = settings["ann_analog_input_l2"]
        else:
            input_l2 = 0.0

        x = tf.keras.layers.Dense(
            nodes, activation=settings["ann_analog_model_act"],
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+1),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=input_l2),
        )(x)

    # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+2),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+3),
    )(x) #could add in a kernel constraint to make the weight positive

    prediction_model = tf.keras.Model(
        inputs=[soi_input, analog_input],
        outputs=[prediction],
        name='prediction_model'
    )

    return prediction_model


def build_ann_model(settings, x_train):

    map_dim = x_train[0].shape[1:]
    # placeholder is not used, just here to stay consistent with other model architectures
    placeholder = tf.keras.layers.Input(shape=map_dim, name='placeholder')
    analog_input = tf.keras.layers.Input(shape=map_dim, name='soi_input')

    # Build Model
    x = tf.keras.layers.Flatten()(analog_input)

    # Add all the Dense Layers
    for layer, nodes in enumerate(settings["ann_model_nodes"]):

        if layer == 0:
            input_l2 = settings["ann_input_l2"]
        else:
            input_l2 = 0.0
        x = tf.keras.layers.Dense(
            nodes, activation=settings["ann_model_act"],
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+1),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=input_l2),
        )(x)

    # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+2),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+3),
    )(x)

    prediction_model = tf.keras.Model(
        inputs=[placeholder, analog_input],
        outputs=[prediction],
        name='prediction_model'
    )

    return prediction_model

