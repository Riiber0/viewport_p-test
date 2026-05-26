import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

def sin_convert(pitch_pred_sin, yaw_pred_sin):
    pitch_rad = math.asin(max(-1.0, min(1.0, pitch_pred_sin)))
    yaw_rad = math.asin(max(-1.0, min(1.0, yaw_pred_sin)))

    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)

    return pitch, yaw

def get_viewport_tiles(pitch_sin, pitch_cos, yaw_sin, yaw_cos):
    GRID_COLS = 20
    GRID_ROWS = 10
    HFOV = 100.0  
    ASPECT_RATIO = 16/9
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    VFOV = HFOV / ASPECT_RATIO  
    
    #pitch = math.degrees(math.atan2(pitch_sin, pitch_cos))
    #yaw = math.degrees(math.atan2(yaw_sin, yaw_cos))
    pitch =  math.degrees(pitch_sin)
    yaw = math.degrees(yaw_sin)

    # calculate viewport boundaries
    yaw_min = yaw - (HFOV / 2)
    yaw_max = yaw + (HFOV / 2)
    pitch_min = pitch - (VFOV / 2)
    pitch_max = pitch + (VFOV / 2)
    
    # pitch range
    pitch_min = max(pitch_min, -90.0)
    pitch_max = min(pitch_max, 90.0)
    
    viewport_segments = []
    
    if yaw_min < -180 and yaw_max <= 180:
        viewport_segments.append((yaw_min + 360, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max, pitch_min, pitch_max))
    elif yaw_max > 180 and yaw_min >= -180:
        viewport_segments.append((yaw_min, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max - 360, pitch_min, pitch_max))
    elif yaw_min < -180 and yaw_max > 180:
        viewport_segments.append((-180.0, 180.0, pitch_min, pitch_max))
    else:
        viewport_segments.append((yaw_min, yaw_max, pitch_min, pitch_max))
    
    tiles = list()
    
    for y_min, y_max, p_min, p_max in viewport_segments:
        u_min = (y_min + 180.0) / 360.0
        u_max = (y_max + 180.0) / 360.0
        v_min = (p_min + 90.0) / 180.0
        v_max = (p_max + 90.0) / 180.0
        
        u_min = max(0.0, min(1.0, u_min))
        u_max = max(0.0, min(1.0, u_max))
        v_min = max(0.0, min(1.0, v_min))
        v_max = max(0.0, min(1.0, v_max))
        
        col_min = int(u_min * GRID_COLS)
        col_max = int((u_max - 1e-6) * GRID_COLS)
        row_min = int(v_min * GRID_ROWS)
        row_max = int((v_max - 1e-6) * GRID_ROWS)
        
        col_min = max(0, col_min)
        col_max = min(GRID_COLS - 1, col_max)
        row_min = max(0, row_min)
        row_max = min(GRID_ROWS - 1, row_max)
        
        for col in range(col_min, col_max + 1):
            for row in range(row_min, row_max + 1):
                tiles.append(row * GRID_COLS + col + 1)

    
    return sorted(tiles)

def get_viewport_tiles_sin(pitch_pred_sin, yaw_pred_sin):
    GRID_COLS = 20
    GRID_ROWS = 10
    HFOV = 100.0  
    ASPECT_RATIO = 16/9
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    VFOV = HFOV / ASPECT_RATIO  
    
    # convert sin to degrees
    pitch_rad = math.asin(max(-1.0, min(1.0, pitch_pred_sin)))
    yaw_rad = math.asin(max(-1.0, min(1.0, yaw_pred_sin)))
    
    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)
    
    # calculate viewport boundaries
    yaw_min = yaw - (HFOV / 2)
    yaw_max = yaw + (HFOV / 2)
    pitch_min = pitch - (VFOV / 2)
    pitch_max = pitch + (VFOV / 2)
    
    # pitch range
    pitch_min = max(pitch_min, -90.0)
    pitch_max = min(pitch_max, 90.0)
    
    viewport_segments = []
    
    if yaw_min < -180 and yaw_max <= 180:
        viewport_segments.append((yaw_min + 360, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max, pitch_min, pitch_max))
    elif yaw_max > 180 and yaw_min >= -180:
        viewport_segments.append((yaw_min, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max - 360, pitch_min, pitch_max))
    elif yaw_min < -180 and yaw_max > 180:
        viewport_segments.append((-180.0, 180.0, pitch_min, pitch_max))
    else:
        viewport_segments.append((yaw_min, yaw_max, pitch_min, pitch_max))
    
    tiles = list()
    
    for y_min, y_max, p_min, p_max in viewport_segments:
        u_min = (y_min + 180.0) / 360.0
        u_max = (y_max + 180.0) / 360.0
        v_min = (p_min + 90.0) / 180.0
        v_max = (p_max + 90.0) / 180.0
        
        u_min = max(0.0, min(1.0, u_min))
        u_max = max(0.0, min(1.0, u_max))
        v_min = max(0.0, min(1.0, v_min))
        v_max = max(0.0, min(1.0, v_max))
        
        col_min = int(u_min * GRID_COLS)
        col_max = int((u_max - 1e-6) * GRID_COLS)
        row_min = int(v_min * GRID_ROWS)
        row_max = int((v_max - 1e-6) * GRID_ROWS)
        
        col_min = max(0, col_min)
        col_max = min(GRID_COLS - 1, col_max)
        row_min = max(0, row_min)
        row_max = min(GRID_ROWS - 1, row_max)
        
        for col in range(col_min, col_max + 1):
            for row in range(row_min, row_max + 1):
                tiles.append(row * GRID_COLS + col + 1)

    
    return sorted(tiles)

def get_viewport_tiles_rad(pitch_rad, yaw_rad):

    GRID_COLS = 20
    GRID_ROWS = 10

    HFOV = 100.0
    ASPECT_RATIO = 16 / 9

    VFOV = HFOV / ASPECT_RATIO

    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)

    yaw_min = yaw - (HFOV / 2)
    yaw_max = yaw + (HFOV / 2)

    pitch_min = pitch - (VFOV / 2)
    pitch_max = pitch + (VFOV / 2)

    pitch_min = max(pitch_min, -90.0)
    pitch_max = min(pitch_max, 90.0)

    viewport_segments = []

    if yaw_min < -180 and yaw_max <= 180:
        viewport_segments.append((yaw_min + 360, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max, pitch_min, pitch_max))

    elif yaw_max > 180 and yaw_min >= -180:
        viewport_segments.append((yaw_min, 180.0, pitch_min, pitch_max))
        viewport_segments.append((-180.0, yaw_max - 360, pitch_min, pitch_max))

    elif yaw_min < -180 and yaw_max > 180:
        viewport_segments.append((-180.0, 180.0, pitch_min, pitch_max))
    else:
        viewport_segments.append((yaw_min, yaw_max, pitch_min, pitch_max))

    tiles = []

    for y_min, y_max, p_min, p_max in viewport_segments:

        u_min = (y_min + 180.0) / 360.0
        u_max = (y_max + 180.0) / 360.0

        v_min = (p_min + 90.0) / 180.0
        v_max = (p_max + 90.0) / 180.0

        u_min = max(0.0, min(1.0, u_min))
        u_max = max(0.0, min(1.0, u_max))

        v_min = max(0.0, min(1.0, v_min))
        v_max = max(0.0, min(1.0, v_max))

        col_min = int(u_min * GRID_COLS)
        col_max = int((u_max - 1e-6) * GRID_COLS)

        row_min = int(v_min * GRID_ROWS)
        row_max = int((v_max - 1e-6) * GRID_ROWS)

        col_min = max(0, col_min)
        col_max = min(GRID_COLS - 1, col_max)

        row_min = max(0, row_min)
        row_max = min(GRID_ROWS - 1, row_max)

        for col in range(col_min, col_max + 1):
            for row in range(row_min, row_max + 1):

                tile_id = row * GRID_COLS + col + 1

                tiles.append(tile_id)

    return sorted(set(tiles))

def prepare_data(df, target_pitch, target_yaw):

    df = df.copy()

    df['pitch_sin'] = np.sin(df['pitch'])
    df['pitch_cos'] = np.cos(df['pitch'])
    df['yaw_sin'] = np.sin(df['yaw'])
    df['yaw_cos'] = np.cos(df['yaw'])

    velocity_scaler = StandardScaler()
    velocities = df[['pitch_v', 'yaw_v']].values
    velocities_scaled = velocity_scaler.fit_transform(velocities)

    df['pitch_v_scaled'] = velocities_scaled[:, 0]
    df['yaw_v_scaled'] = velocities_scaled[:, 1]

    df['pitch_pred_sin'] = np.sin(df[target_pitch])
    df['pitch_pred_cos'] = np.cos(df[target_pitch])
    df['yaw_pred_sin'] = np.sin(df[target_yaw])
    df['yaw_pred_cos'] = np.cos(df[target_yaw])

    return df


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if train_ratio+val_ratio+test_ratio != 1:
        print('ratio not equal to 1')
        return

    id_cols = ['v_id', 'u_id', 'playback_time']

    features_cols = ['pitch', 'yaw', 'pitch_v', 'yaw_v']
    
    target_cols = ['pitch_pred_1', 'yaw_pred_1', 
                    'pitch_pred_2', 'yaw_pred_2',
                    'pitch_pred_3', 'yaw_pred_3',
                    'pitch_pred_4', 'yaw_pred_4',
                    'pitch_pred_5', 'yaw_pred_5']

    new_df = df[id_cols + features_cols + target_cols].copy()

    train_data, val_data, test_data = [], [], []

    for v_id in new_df['v_id'].unique():
        video_df = new_df[new_df['v_id'] == v_id].copy()

        users = video_df['u_id'].unique()
        np.random.shuffle(users)
        n_users = len(users)

        n_train = max(1, int(n_users * train_ratio))
        n_val = max(1, int(n_users * val_ratio))
        n_test = n_users - n_train - n_val

        train_users = users[:n_train]
        val_users = users[n_train:n_train + n_val]
        test_users = users[n_train + n_val:]

        train_data.append(video_df[video_df['u_id'].isin(train_users)])
        val_data.append(video_df[video_df['u_id'].isin(val_users)])
        test_data.append(video_df[video_df['u_id'].isin(test_users)])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df

def create_temporal_sequences(df, pred_time, use_v=True):
    df = df.sort_values(['v_id', 'u_id', 'playback_time']).reset_index(drop=True)

    if use_v:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos', 
                         'pitch_v_scaled', 'yaw_v_scaled']
    else:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos']

    target_cols = ['pitch_pred_sin', 'pitch_pred_cos', 
                   'yaw_pred_sin', 'yaw_pred_cos']

    #seq_len = 40 - pred_time
    seq_len = 35
    
    x_sequences = []
    y_targets = []

    sessions = df.groupby(['v_id', 'u_id'])

    for (v_id, u_id), session_df in sessions:
        features = session_df[features_cols].values
        target = session_df[target_cols].values

        target_idx = seq_len
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len-1]
            target_idx = i + seq_len - 1

            x_sequences.append(seq)
            y_targets.append(target[target_idx])

    x_seq = np.array(x_sequences, dtype=np.float64)
    y_seq = np.array(y_targets, dtype=np.float64)

    return x_seq, y_seq

def create_detailed_temporal_sequences(df, pred_time, use_v=True):
    df = df.sort_values(['v_id', 'u_id', 'playback_time']).reset_index(drop=True)

    if use_v:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos', 
                         'pitch_v_scaled', 'yaw_v_scaled']
    else:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos']

    target_cols = ['pitch_pred_sin', 'pitch_pred_cos', 
                   'yaw_pred_sin', 'yaw_pred_cos']

    seq_len = 40 - pred_time
    
    tests = []

    sessions = df.groupby(['v_id', 'u_id'])

    for (v_id, u_id), session_df in sessions:
        features = session_df[features_cols].values
        target = session_df[target_cols].values
        playback_times = session_df['playback_time'].values

        target_idx = seq_len
        for i in range(len(features) - seq_len):
            if i-seq_len+2 > 0:
                #seq = features[i:i+seq_len-1]
                #seq = features[i-seq_len+1:i]
                #seq = np.array(seq, dtype=np.float64)
                seq = True
            else:
                seq = False
                #seq = None

            target_idx = i #+ seq_len - 1

            tests.append({'v_id': v_id,
                                'u_id': u_id,
                                'playback_time': playback_times[target_idx],
                                'sequence': seq,
                                'target': target[target_idx]
                })

    #x_seq = np.array(x_sequences, dtype=np.float64)
    ret_df = pd.DataFrame(tests)

    return ret_df

def prepare_test_data(df):
    test_data = []
    for index, row in df.iterrows():
        tiles = get_viewport_tiles_rad(row['pitch'], row['yaw'])
        new_row = {'v_id': row['v_id'],
                   'u_id': row['u_id'],
                   'playback_time': row['playback_time'],
                   'tiles': tiles
                   }
        test_data.append(new_row)

    return pd.DataFrame(test_data)

if __name__ == '__main__':
    df = pd.read_csv('dataset_processed.csv')
    df = df[df['v_id'] == 1]
    users = df['u_id'].unique()
    user = users[0]

    df = df[df['u_id'] == user]

    df = df.sort_values('playback_time')

    test_data = prepare_test_data(df)
    print(test_data)

