import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, GRU, LSTM, MaxPooling1D, Dropout, TimeDistributed, Flatten, Dense, Bidirectional, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from custom_keras import *
import sys
import math

MODELS_PATH='./models/'


class predModel():
    def __init__(self, name):
        self.model = None
        self.name = name

    @abstractmethod
    def create_regression_model(self, input_shape):
        pass

    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.AdamW(), loss=AngularLoss(), 
                           metrics=[JaccardMetric(), OrthodromicMetric()])

    def model_summary(self):
        self.model.summary()

    def train_model(self, x_train, y_train, x_val, y_val, b_size):
        """
        callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(patience=10, verbose=1),
                ModelCheckpoint(MODELS_PATH+self.name+'kfix_model.h5',save_best_only=True, verbose=1),
                CSVLogger(MODELS_PATH+self.name+'kfix_training.log', separator=',', append=False)
        ]
        """
        callbacks = [
                ReduceLROnPlateau(patience=10, verbose=1),
                SaveAfterEpoch(start_epoch=-1, filepath = MODELS_PATH+self.name+'_model_{epoch}.h5'),
                CSVLogger(MODELS_PATH+self.name+'_training.log', separator=',', append=True)
        ]

        #last_epoch = self.check_model()
        last_epoch = 0
        #self.model = load_model(MODELS_PATH+self.name+'_kmov_model_10.h5')

        hist = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, 
                              batch_size=b_size, 
                              callbacks=callbacks, verbose=1,
                              initial_epoch=last_epoch)

        return hist

    def test_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def plot_model(self):
        keras.utils.plot_model(self.model, to_file=self.name)

    def check_model(self):
        epoch = 10
        path_name = MODELS_PATH+self.name+'kfix_model_{epoch}.h5'.format(epoch = epoch)
        f = Path(path_name)
        while f.is_file():
            epoch += 1
            path_name = MODELS_PATH+self.name+'kfix_model_{epoch}.h5'.format(epoch = epoch)
            #print(path_name)
            f = Path(path_name)

        path_name = MODELS_PATH+self.name+'kfix_model_{epoch}.h5'.format(epoch = epoch - 1)
        self.model = load_model(path_name)
        return epoch - 1

class GRUpred(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        g1 = layers.GRU(32, return_sequences=True, dropout=0.1)(input_layer)
        g1 = layers.LayerNormalization()(g1)

        g2 = layers.GRU(16)(g1)

        d3 = layers.Dense(16, activation="tanh")(g2)

        output_layer = Dense(4)(d3)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model
        
class LSTMpred(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        l1 = layers.LSTM(32, return_sequences=True, dropout=0.1)(input_layer)
        l1 = layers.LayerNormalization()(l1)

        l2 = layers.LSTM(16)(l1)

        d3 = layers.Dense(16, activation="tanh")(l2)

        output_layer = layers.Dense(4)(d3)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

class CNNpred(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        c1 = layers.Conv1D(16, kernel_size=3, padding="causal",)(input_layer)
        c1 = layers.LayerNormalization()(c1)
        c1 = layers.Activation("gelu")(c1)

        c2 = layers.Conv1D(16, kernel_size=3, dilation_rate=2, padding="causal",)(c1)
        c2 = layers.LayerNormalization()(c2)
        c2 = layers.Activation("gelu")(c2)
        c2 = layers.GlobalAveragePooling1D()(c2)

        d3 = layers.Dense(16, activation="gelu")(c2)

        output_layer = layers.Dense(4)(d3)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

class TCN(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape = input_shape)
        
        c1 = Conv1D(32, 1)(input_layer)

        r2 = ResidualBlock(32, 1)(c1)
        r2 = ResidualBlock(32, 2)(r2)
        r2 = ResidualBlock(32, 4)(r2)
        r2 = GlobalAveragePooling1D()(r2)

        d3 = Dense(16, activation='gelu')(r2)

        output_layer = Dense(4)(d3)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

class NeuralKalman(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        kalman_pred = KalmanLayer()(input_layer)

        x = GRU(32)(input_layer)

        residual = Dense(16, activation="tanh")(x)

        residual = Dense(4)(residual)

        output_layer = layers.Add()([kalman_pred, residual])

        self.model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

class Transformer(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        x = Dense(32)(input_layer)

        x = TransformerBlock()(x)
        x = TransformerBlock()(x)

        x = GlobalAveragePooling1D()(x)

        x = Dense(16, activation="gelu")(x)

        output_layer = Dense(4)(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

def get_model(model_name, test_shape):
    ret = None

    match model_name:
        case 'CNN':
            ret = CNNpred('CNN')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary()
        case 'GRU':
            ret = GRUpred('GRU')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary()
        case 'LSTM':
            ret = LSTMpred('LSTM')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary()
        case 'TCN':
            ret = TCN('TCN')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary()
        case 'KAL':
            ret = NeuralKalman('NeuralKalman')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary() 
        case 'TPT':
            ret = Transformer('Transformer')
            ret.create_regression_model(test_shape)
            ret.compile_model()
            ret.model_summary()

    return ret

if __name__ == '__main__':
    models = ['CNN', 'GRU', 'LSTM', 'TCN', 'KAL', 'TPT']
    
    for target in range(1, 6):
        pred_time = 0.5*target
        k = math.ceil((pred_time*10)/2)


        train_data = prepare_data(train_df, f'pitch_pred_{target}', f'yaw_pred_{target}')
        val_data = prepare_data(val_df, f'pitch_pred_{target}', f'yaw_pred_{target}')

        x_train, y_train = create_temporal_sequences(train_data, True, k)
        x_val, y_val = create_temporal_sequences(val_data, True, k)

        for model in models:
            shape = (k, 6)
            m = get_model(model, shape)
            m.name = f'{model}_{target}'
            m.train_model(x_train, y_train, x_val, y_val, b_size=512)

#####################################old##########################################

class CNN_GRU(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Bloco de duas Camadas CNN
        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(c1)
        c1 = MaxPooling1D(max_pooling)(c1)
        c1 = Dropout(dropout)(c1)

        #Segunda camada convolucional
        c2 = Conv1D(128, 3, activation='relu', padding='same')(c1)
        c2 = MaxPooling1D(max_pooling)(c2)
        c2 = Dropout(dropout)(c2)

        #Bloco de duas camada GRU
        #Primeira camada GRU
        g3 = Bidirectional(GRU(128, return_sequences=True, dropout=dropout))(c2)
        g3 = BatchNormalization()(g3)

        #Segunda camada GRU
        g4 = Bidirectional(GRU(64, return_sequences=False, dropout=dropout))(g3)
        g4 = BatchNormalization()(g4)

        #Bloco de camada densa para regressao
        d5 = Dense(64, activation='relu')(g4)
        d5 = Dropout(dropout)(d5)

        #Camada de saida
        output_layer = Dense(4)(d5)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model


class baseModel(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(c1)
        c1 = MaxPooling1D(max_pooling)(c1)
        c1 = Dropout(dropout)(c1)

        #Flatten
        f2 = Flatten()(c1)

        #Bloco de camada densa para regressao
        d3 = Dense(64, activation='relu')(f2)
        d3 = Dropout(dropout)(d3)

        #Camada de saida
        output_layer = Dense(4)(d3)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

class CNN_LSTM(predModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Bloco de duas Camadas CNN
        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(c1)
        c1 = MaxPooling1D(max_pooling)(c1)
        c1 = Dropout(dropout)(c1)

        #Segunda camada convolucional
        c2 = Conv1D(128, 3, activation='relu', padding='same')(c1)
        c2 = MaxPooling1D(max_pooling)(c2)
        c2 = Dropout(dropout)(c2)

        #Bloco de duas camada LSTM
        #Primeira camada LSTM
        g3 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout))(c2)
        g3 = BatchNormalization()(g3)

        #Segunda camada LSTM
        g4 = Bidirectional(LSTM(64, return_sequences=False, dropout=dropout))(g3)
        g4 = BatchNormalization()(g4)

        #Bloco de camada densa para regressao
        d5 = Dense(64, activation='relu')(g4)
        d5 = Dropout(dropout)(d5)

        #Camada de saida
        output_layer = Dense(4)(d5)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

