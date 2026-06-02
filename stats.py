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
from get_preds import get_grap_pred, get_flare_pred, get_model_pred, get_static_pred

GRAPH_PATH = "./graphs/"

def calculate_orthodromic_distance(new_df):
    error = 1 - (np.sin(new_df['pred_pitch'])*np.sin(new_df['pitch'])+np.cos(new_df['pred_pitch']) * np.cos(new_df['pitch']) * np.cos( new_df['pred_yaw'] - new_df['yaw']))

    mean_error = np.mean(error)

    confidence = 0.95
    n = len(error)

    se = stats.sem(error)

    ci_lower, ci_upper = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean_error,
        scale=se
    )

    return [mean_error, ci_lower, ci_upper]

def get_mae(pred):
    aggregate_error = (
        np.abs(new_df['pitch'] - new_df['pred_pitch']) +
        np.abs(new_df['yaw'] - new_df['pred_yaw'])
    ) / 2

    aggregate_mae = np.mean(aggregate_error)

    confidence = 0.95
    n = len(aggregate_error)

    se = stats.sem(aggregate_error)

    ci_lower, ci_upper = stats.t.interval(
        confidence,
        df=n - 1,
        loc=aggregate_mae,
        scale=se
    )

    print(f'MAE: {aggregate_mae}')
    print(f'95% CI: [{ci_lower}, {ci_upper}]')

def get_rmse(new_df):
    aggregate_squared_error = (
        (
            new_df['pitch'] - new_df['pred_pitch']
        ) ** 2
        +
        (
            new_df['yaw'] - new_df['pred_yaw']
        ) ** 2
    ) / 2

    aggregate_rmse = np.sqrt(
        np.mean(aggregate_squared_error)
    )

    confidence = 0.95
    n = len(aggregate_squared_error)

    se = stats.sem(aggregate_squared_error)

    mean_squared_error = np.mean(
        aggregate_squared_error
    )

    ci_lower_mse, ci_upper_mse = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean_squared_error,
        scale=se
    )

    ci_lower_rmse = np.sqrt(ci_lower_mse)
    ci_upper_rmse = np.sqrt(ci_upper_mse)

    print(f'RMSE: {aggregate_rmse}')
    print(f'95% CI: [{ci_lower_rmse}, {ci_upper_rmse}]')

def get_jaccard(new_df):
    pred_tiles = new_df['pred_tiles']
    target_tiles = new_df['tiles']

    weighted_jaccard = []

    for pred, target in zip(pred_tiles, target_tiles):

        pred_set = set(pred)
        target_set = set(target)

        intersection = len(pred_set & target_set)
        union = len(pred_set | target_set)

        if union == 0:
            weighted_jaccard.append(0)
        else:
            weighted_jaccard.append(intersection / union)

    weighted_jaccard = np.array(weighted_jaccard)

    mean_wji = np.mean(weighted_jaccard)

    confidence = 0.95
    n = len(weighted_jaccard)

    se = stats.sem(weighted_jaccard)

    ci_lower, ci_upper = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean_wji,
        scale=se
    )

    return [mean_wji, ci_lower, ci_upper]

if __name__ == '__main__':
    test_df = pd.read_csv('test_data.csv')
    test_data = prepare_test_data(test_df)

    pred_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    result = []

    """
    for pw in pred_times:
        predictions = get_grap_pred(pw, test_df)
        new_df = pd.merge(test_data, predictions, on=['v_id', 'u_id', 'playback_time'])
        print(f'navigation graph {pw}:')
        get_jaccard(new_df)
        calculate_orthodromic_distance(new_df)
    """

    for pw in pred_times:
        predictions = get_flare_pred(pw, test_df, True)
        new_df = pd.merge(test_data, predictions, on=['v_id', 'u_id', 'playback_time'])
        jac = get_jaccard(new_df)
        ang = calculate_orthodromic_distance(new_df)
        result.append(
                {'model': 'ridge regression',
                 'pw': pw,
                 'jaccard': jac[0],
                 'jaccard_lower': jac[1],
                 'jaccard_upper': jac[2],
                 'agular_error': ang[0],
                 'angular_lower': ang[1],
                 'angular_upper': ang[2]
                    }
                )

        predictions = get_flare_pred(pw, test_df, False)
        new_df = pd.merge(test_data, predictions, on=['v_id', 'u_id', 'playback_time'])
        jac = get_jaccard(new_df)
        ang = calculate_orthodromic_distance(new_df)
        result.append(
                {'model': 'linear regression',
                 'pw': pw,
                 'jaccard': jac[0],
                 'jaccard_lower': jac[1],
                 'jaccard_upper': jac[2],
                 'agular_error': ang[0],
                 'angular_lower': ang[1],
                 'angular_upper': ang[2]
                    }
                )

    model_files = sorted(glob.glob('models_all/*.h5'))
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.h5', '')
        pw = float(int(model_name.split('_')[2]) * 0.5)
        model_name = model_name.split('_')[0] + '_' + model_name.split('_')[1]

        use_v = True
        kmov = False

        if 'BASE' in model_name.upper():
            use_v = False

        if 'KMOV' in model_name.upper():
            kmov = True

        predictions = get_model_pred(pw, test_df, model_path, use_v, kmov)

        predictions['playback_time'] = predictions['playback_time'] + pw 
        new_df = pd.merge(test_data, predictions, on=['v_id', 'u_id', 'playback_time'])
        jac = get_jaccard(new_df)
        ang = calculate_orthodromic_distance(new_df)
        result.append(
                {'model': model_name,
                 'pw': pw,
                 'jaccard': jac[0],
                 'jaccard_lower': jac[1],
                 'jaccard_upper': jac[2],
                 'agular_error': ang[0],
                 'angular_lower': ang[1],
                 'angular_upper': ang[2]
                }
                )

    result_df = pd.DataFrame(result)
    result_df.to_csv('stats.csv')

