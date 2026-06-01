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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

GRAPH_PATH = "./graphs/"
MODELS_PATH = './models/'
TILE_RESULT_PATH = './tile_result/'

def test_tile_miss(test_data, predictions, pred_time, entropy_df, model_name):
    """
        test_data
    """

    results = []

    end = len(test_data)
    cur = 0
    widgets = [
    'Progress: ',                               
    progressbar.Counter(format='%(value)d/%(max_value)d'),
    ' ',
    progressbar.Bar(marker=':'),      
    ' ',
    progressbar.ETA(),                    
    ' | ',
    progressbar.Timer(),                  
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=end, end='\n')

    model_sum = []
    seg_time = pred_time
    segments_num = int(60/seg_time)
    for (v_id, u_id), session in test_data.groupby(['v_id', 'u_id']):
        session_preds = predictions[
            (predictions['v_id'] == v_id) &
            (predictions['u_id'] == u_id)
        ]
        video_buffer = {}
        segment_tiles = {}

        pred_tiles = []
        total_tile_miss = 0
        session_sum = {}

        for i in range(1, segments_num + 1):
            session_sum['tile_miss_'+str(i)] = 0
            video_buffer[i] = []
            segment_tiles[i] = []

        video_buffer[1] = list(range(1, 201))

        trace_class = entropy_df[(entropy_df['u_id'] == u_id) & (entropy_df['v_id'] == v_id)]['classificacao']
        #print(f"v_id:{v_id}, u_id:{u_id}")
        #print(trace_class)
        #print(trace_class.iloc[0])
        trace_class = trace_class.iloc[0]

        for index, row in session.iterrows():
            pred_row = session_preds[np.isclose(
                session_preds['playback_time'], 
                row['playback_time'], atol=1e-6)].head(1)
            tile_miss = 0
            download_size = 0
            fov_tiles = 0
            wasted_tiles = 0
            #if row['sequence']:
                #pred = m.predict(np.array([row['sequence']]), batch_size=1, verbose=0)[0]
                #print(f"{pred} - {predictions[pred_index]}")
                #pred = predictions[pred_index]
                #pred_tiles = get_viewport_tiles(pred[0], pred[1], pred[2], pred[3])

            current_seg = int(np.floor(row['playback_time'] / seg_time)) + 1
            pred_seg = int(np.floor((row['playback_time'] + pred_time) / seg_time)) + 1

            if not pred_row.empty:
                pred_tiles = pred_row['pred_tiles'].iloc[0]
            else:
                pred_tiles = []

            if pred_seg > segments_num:
                tiles_to_download = []
            else:
                tiles_to_download = [tile for tile in pred_tiles if tile not in video_buffer[pred_seg]]
                video_buffer[pred_seg] += tiles_to_download
                video_buffer[pred_seg].sort()

            download_size = len(tiles_to_download)


            #print(f"{index}: current segment = {current_seg}, playback time = {row['playback_time']}")
            #print(get_viewport_tiles(row['target'][0], row['target'][2]))
            #print(video_buffer[current_seg])

            """mudado para tiles"""
            #view_port_tiles = get_viewport_tiles(row['target'][0], row['target'][1], 
            #        row['target'][2], row['target'][3])
            view_port_tiles = row['tiles']

            fov_tiles = len(view_port_tiles)
            segment_tiles[current_seg] += list(set(view_port_tiles) - set(segment_tiles[current_seg]))
            """
            if row['playback_time'] == 4.0:
                print('====================')
                print(current_seg)
                print(segment_tiles[current_seg])
                print(video_buffer[current_seg])
                print(set(video_buffer[current_seg]) - set(segment_tiles[current_seg]))
                sys.exit()
            """

            if not set(view_port_tiles).issubset(set(video_buffer[current_seg])):
                #print(set(view_port_tiles) - set(video_buffer[current_seg]))
                #print(view_port_tiles)
                #print(video_buffer[current_seg])

                missed_tiles = set(view_port_tiles) - set(video_buffer[current_seg])
                total_tile_miss += len(missed_tiles)
                session_sum['tile_miss_'+str(current_seg)] += len(missed_tiles)

                tile_miss = len(missed_tiles)

                video_buffer[current_seg] = list(set(video_buffer[current_seg]) | missed_tiles)

            wasted_tiles = len(set(video_buffer[current_seg]) - set(segment_tiles[current_seg]))

            line_result = {'u_id': u_id,
                            'v_id': v_id,
                            'trace_class': trace_class,
                            'playback_time': row['playback_time'],
                            'segment': current_seg,
                            'download_size': download_size,
                            'tile_miss': tile_miss,
                            'fov_tiles': fov_tiles,
                            'wasted_tiles': wasted_tiles
                    }

            model_sum.append(line_result)

            cur += 1
            bar.update(cur)
        
        session_sum['u_id'] = u_id
        session_sum['v_id'] = v_id
        session_sum['total_tile_miss'] = total_tile_miss
        session_sum['model'] = model_name
        session_sum['pred_time'] = pred_time
        session_sum['trace_class'] = trace_class
                        
        results.append(session_sum)

    model_df = pd.DataFrame(model_sum)
    #model_df.to_csv(TILE_RESULT_PATH+model_name+str(int(pred_time*10))+'.csv')
    model_df.to_csv(f'{TILE_RESULT_PATH}{model_name}_{int(pred_time*10)}.csv')

    bar.finish()
    return results

if __name__ == '__main__':
    test_df = pd.read_csv('test_data.csv')
    entropy_df = pd.read_csv('entropy_classified.csv')

    test_data = prepare_test_data(test_df)


    pred_times = [0.5, 1.0, 1.5, 2.0, 2.5]

    for pw in pred_times:
        predictions = get_static_pred(test_df)
        test_tile_miss(test_data, predictions, pw, entropy_df, 'static_pred')

    for pw in pred_times:
        predictions = get_grap_pred(pw, test_df)
        test_tile_miss(test_data, predictions, pw, entropy_df, 'graph_navagation')

    for pw in pred_times:
        predictions = get_flare_pred(pw, test_df, True)
        test_tile_miss(test_data, predictions, pw, entropy_df, 'linear_regression')

        predictions = get_flare_pred(pw, test_df, False)
        test_tile_miss(test_data, predictions, pw, entropy_df, 'ridge_regression')

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

        print(f'{model_name} | {pw} | {use_v}')

        predictions = get_model_pred(pw, test_df, model_path, use_v, kmov)
        test_tile_miss(test_data, predictions, pw, entropy_df, model_name)

