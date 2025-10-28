# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% imports
import keras
import numpy as np


# %% generating datapoints from Normal distribution
# generate a dataset with 10000 data points drawn from a gaussian
# distribution. Our dataset will have 5 features, but the rest will be
# correlated/ duplicated

normal_samples = np.random.normal(loc=5, scale=1, size=(10000, 5))

correlated_samples = []
for i in range(3):
    feature = np.random.randint(1, 5, dtype=int)
    correlated_samples.append(
        (normal_samples[:, feature] * np.random.random())
        + np.random.normal(loc=0, scale=0.2, size=10000)
    )

redundant_samples = normal_samples[:, np.random.randint(1, 5, size=2)]

classA_samples = np.hstack(
    [normal_samples, redundant_samples, np.array(correlated_samples).T]
)

# %% Generating data points from Poisson distribution

# Generate another dataset with 10000 data points, but this is drawn from
# poisson distribution
poisson_samples = np.random.poisson(5, size=(10000, 5))
# poisson_samples = np.random.normal(loc=50, scale=3, size=(10000, 5))

correlated_poisson_samples = []
for i in range(3):
    # randomly select a feature to which we will do a linear transformation
    # and add it as new feature
    feature = np.random.randint(1, 5, dtype=int)
    correlated_poisson_samples.append(
        (poisson_samples[:, feature] * np.random.random())
        + np.random.normal(loc=0, scale=0.2, size=10000)
    )

correlated_poisson_samples = np.array(correlated_poisson_samples).T

# add two more duplicated samples
redundant_poisson_samples = poisson_samples[:, np.random.randint(1, 5, size=2)]

classB_samples = np.hstack(
    [poisson_samples, correlated_poisson_samples, redundant_poisson_samples]
)


# %% Defining an auto-encoder model
# define an autoencoder model
def auto_encoder():
    input_layer = keras.layers.Input(shape=(10,))
    encoder_layer1 = keras.layers.Dense(8, activation="relu")(input_layer)
    encoder_layer2 = keras.layers.Dense(6, activation="relu")(encoder_layer1)
    coder_layer = keras.layers.Dense(5, activation="relu")(encoder_layer2)
    decoder_layer1 = keras.layers.Dense(7, activation="relu")(coder_layer)
    decoder_layer2 = keras.layers.Dense(9, activation="relu")(decoder_layer1)
    output_layer = keras.layers.Dense(10, activation="relu")(decoder_layer2)

    model = keras.Model(input_layer, output_layer)

    return model


classA_encoder = auto_encoder()
# classB_encoder = keras.models.clone_model(classA_encoder, )
classB_encoder = auto_encoder()

# %% Fitting the model to both the class of data points
classA_encoder.compile(optimizer="adam", loss="mse")
classA_encoder_history = classA_encoder.fit(
    x=classA_samples, y=classA_samples, epochs=1000, validation_split=0.2
)

# %% Fitting classB encoder
classB_encoder.compile(optimizer="adam", loss="mse")
classB_encoder_history = classB_encoder.fit(
    x=classB_samples, y=classB_samples, epochs=1000, validation_split=0.2
)

# %% evaluating classA_encoder
# evaluate the model using same and different class examples
inclass_loss = classA_encoder.evaluate(classA_samples, classA_samples)
outofclass_loss = classA_encoder.evaluate(classB_samples, classB_samples)

print(inclass_loss, outofclass_loss)
# 0.5513802766799927 18.516738891601562

# %% evaluating classB_encoder
inclass_bloss = classB_encoder.evaluate(classB_samples, classB_samples)
outofclass_bloss = classB_encoder.evaluate(classA_samples, classA_samples)

print(inclass_bloss, outofclass_bloss)
# 18.672155380249023 1.9210742712020874
# %%
