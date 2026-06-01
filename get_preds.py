import numpy as np
import pandas as pd
import os, sys
import progressbar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '2' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import R2Score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from utils import *
from model import CNN_GRU, baseModel, CNN_LSTM
import glob
from graph_navagation import NavigationGraphPredictor
from flare import FlareLinearRegressionPredictor, FlareRidgeRegressionPredictor

GRAPH_PATH = "./graphs/"
MODELS_PATH = "./models/"

def get_grap_pred(pred_time, test_df):
    pw = int(pred_time*10)
    predictions = []
    for (v_id, u_id), session in test_df.groupby(['v_id', 'u_id']):
        
        predictor = NavigationGraphPredictor(pw, GRAPH_PATH+f'video_{v_id}_{pw}_ng.xml')

        for _, row in session.iterrows():
            pitch = float(row['pitch'])
            yaw = float(row['yaw'])

            viewport_tiles = get_viewport_tiles_rad(
                pitch_rad=pitch,
                yaw_rad=yaw
            )

            pred = predictor.predict(viewport_tiles)
            new_row = {'v_id': row['v_id'],
                       'u_id': row['u_id'],
                       'playback_time': row['playback_time'],
                       'pred_tiles': pred
                       }
            predictions.append(new_row)

    return pd.DataFrame(predictions)

def get_flare_pred(pred_time, test_df, linear):

    prediction_points = int(pred_time / 0.1)

    predictions = []

    for (v_id, u_id), session in test_df.groupby(['v_id', 'u_id']):

        if linear:
            predictor = FlareLinearRegressionPredictor(history_window=5, prediction_points=prediction_points)
        else:
            predictor = FlareRidgeRegressionPredictor(history_window=5,prediction_points=prediction_points)

        for _, row in session.iterrows():

            pitch_deg, yaw_deg = get_degree_from_rad(row['pitch'], row['yaw'])

            predictor.add_sample(yaw=yaw_deg, pitch=pitch_deg)

            pred = predictor.predict()

            if pred is None:
                pred = None

            else:

                yaw_pred_deg, pitch_pred_deg = pred
                pitch_rad = math.radians(pitch_pred_deg) 
                yaw_rad = math.radians(yaw_pred_deg)

                pred_tiles = get_viewport_tiles(pitch_pred_deg, yaw_pred_deg)

                new_row = {
                    'v_id': row['v_id'],
                    'u_id': row['u_id'],
                    'playback_time': row['playback_time'],
                    'pred_pitch': pitch_rad,
                    'pred_yaw': yaw_rad,
                    'pred_tiles': pred_tiles
                }

                predictions.append(new_row)

    return pd.DataFrame(predictions)

def get_model_pred(pred_time, test_df, model_path, use_v, kmov):
    pred_id = int(pred_time * 10)
    target_id = int(pred_id/5)

    m = load_model(model_path, compile = False)
    

    test_data = prepare_data(test_df, f'pitch_pred_{target_id}', f'yaw_pred_{target_id}')
    x_test, y_test = create_temporal_sequences(test_data, pred_id, use_v, kmov)
    coord_predictions = m.predict(x_test)
    pred_index = 0
    predictions = []
    for (v_id, u_id), session in test_df.groupby(['v_id', 'u_id']):

        for _, row in session.iterrows():
            if row['playback_time'] < 3.5:
                pred_tiles = []
                pitch_rad = 0
                yaw_rad = 0
            else:
                pred = coord_predictions[pred_index]

                pitch_rad = math.atan2(pred[0], pred[1])
                yaw_rad = math.atan2(pred[2], pred[3])

                pred_tiles = get_viewport_tiles_rad(pitch_rad, yaw_rad)

                new_row = {
                    'v_id': row['v_id'],
                    'u_id': row['u_id'],
                    'playback_time': row['playback_time'],
                    'pred_pitch': pitch_rad,
                    'pred_yaw': yaw_rad,
                    'pred_tiles': pred_tiles
                }

                predictions.append(new_row)
                pred_index += 1

    return pd.DataFrame(predictions)

def get_static_pred(test_df):

    predictions = []
    for (v_id, u_id), session in test_df.groupby(['v_id', 'u_id']):

        for _, row in session.iterrows():

            pred_tiles = get_viewport_tiles_rad(row['pitch'], row['yaw'])

            new_row = {
                'v_id': row['v_id'],
                'u_id': row['u_id'],
                'playback_time': row['playback_time'],
                'pred_pitch': row['pitch'],
                'pred_yaw': row['yaw'],
                'pred_tiles': pred_tiles
            }

            predictions.append(new_row)

    return pd.DataFrame(predictions)
