import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,Multiply,BatchNormalization,
                                     Dropout, Flatten, Dense, Activation, Permute)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd

from models.attention import Attention  

class ConvNet(tf.keras.Model):
    def __init__(self, model_input_shape, n_classes,
                 optimizer=Adam, loss='categorical_crossentropy',
                 model_metrics=['acc'], learning_rate=3e-04):
        super(ConvNet, self).__init__()
        self.model_input_shape = model_input_shape
        self.n_classes = n_classes

        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss = loss
        self.model_metrics = model_metrics

        self.build_model()


    def build_model(self):

        input_layer = Input(shape=self.model_input_shape)

        latent = Conv1D(filters=256, kernel_size=3, activation='relu')(input_layer) 
        latent = BatchNormalization()(latent)
        attention_layer_1 = Attention(d_model=256, d_k=256, d_v=256) 
        latent = attention_layer_1(latent, latent, latent) 

        latent = Conv1D(filters=128, kernel_size=5, activation='relu')(latent) 
        latent = BatchNormalization()(latent) 
        attention_layer_2 = Attention(d_model=128, d_k=128, d_v=128) 
        latent = attention_layer_2(latent, latent, latent) 

        latent = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(latent) 
        latent = BatchNormalization()(latent) 
        attention_layer_3 = Attention(d_model=64, d_k=64, d_v=64) 
        latent = attention_layer_3(latent, latent, latent) 


        latent = Conv1D(filters=32, kernel_size=9, activation='relu', padding='same')(latent)
        latent = BatchNormalization()(latent) 

        
        latent = Flatten()(latent)
        latent = Dense(units=8, activation='relu')(latent)
        # latent = Dropout(rate=0.5)(latent)  
        
        logits_layer = Dense(units=self.n_classes)(latent)
        output_layer = Activation('softmax')(logits_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.logits_model = Model(inputs=input_layer, outputs=logits_layer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.model_metrics)

    def call(self, inputs):
        return self.model(inputs)

    def get_logits(self, x):
        return self.logits_model(x)


    def maybe_train(self, data_train, data_valid, batch_size, epochs):
        DIR_ASSETS = 'assets/'
        PATH_MODEL = DIR_ASSETS + 'teacher_model.hdf5'

        if os.path.exists(PATH_MODEL):
            print('Loading trained model from {}.'.format(PATH_MODEL))
            self.model = load_model(PATH_MODEL, custom_objects={'SFAttention': Attention})
        else:
            print('No checkpoint found on {}. Training from scratch.'.format(PATH_MODEL))
            x_train, y_train = data_train
            x_val, y_val = data_valid

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

            self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                           batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, lr_scheduler])
            print('Saving trained model to {}.'.format(PATH_MODEL))
            if not os.path.isdir(DIR_ASSETS):
                os.mkdir(DIR_ASSETS)
            self.model.save(PATH_MODEL)

    def evaluate(self, x, y):
        if self.model:
            score = self.model.evaluate(x, y)
            print('accuracy: {:.2f}% | loss: {}'.format(100 * score[1], score[0]))
        else:
            print('Missing model instance.')

    def predict(self, x):
        if self.model:
            # return self.model.predict(x)
            predictions = self.model.predict(x)
            return np.argmax(predictions, axis=1) 
        else:
            print('Missing model instance.')

    def predict_on_batch(self, x):
        if self.model:
            return self.model.predict_on_batch(x)
        else:
            print('Missing initialized model instance.')

    def predict_prob(self, x):
        if self.model:
            return self.model.predict(x.reshape(-1, self.input_shape[0], self.input_shape[1]), verbose=1)
        else:
            print('Missing model instance.')




