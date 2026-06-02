import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from utils import get_viewport_tiles_tf 

class AngularLoss(tf.keras.losses.Loss):

    def __init__(self, name="angular_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        y_true = tf.math.l2_normalize(y_true, axis=-1)
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)

        dot = tf.reduce_sum(y_true * y_pred, axis=-1)

        return 1.0 - dot

class JaccardMetric(tf.keras.metrics.Metric):

    def __init__(self, name="jaccard", **kwargs):
        super().__init__(name=name, **kwargs)

        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        pitch_true = tf.atan2(y_true[:,0], y_true[:,1])
        yaw_true   = tf.atan2(y_true[:,2], y_true[:,3])

        pitch_pred = tf.atan2(y_pred[:,0], y_pred[:,1])
        yaw_pred   = tf.atan2(y_pred[:,2], y_pred[:,3])

        true_tiles = get_viewport_tiles_tf(pitch_true, yaw_true)
        pred_tiles = get_viewport_tiles_tf(pitch_pred, yaw_pred)

        intersection = tf.reduce_sum(true_tiles * pred_tiles, axis=1)

        union = (
            tf.reduce_sum(true_tiles, axis=1)
            + tf.reduce_sum(pred_tiles, axis=1)
            - intersection
        )

        jaccard = intersection / (union + 1e-7)

        self.total.assign_add(tf.reduce_sum(jaccard))
        self.count.assign_add(tf.cast(tf.size(jaccard), tf.float32))

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class OrthodromicMetric(tf.keras.metrics.Metric):

    def __init__(self, name="orthodromic", **kwargs):
        super().__init__(name=name, **kwargs)

        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        true_yaw   = y_true[:, 0]
        true_pitch = y_true[:, 1]

        pred_yaw   = y_pred[:, 0]
        pred_pitch = y_pred[:, 1]

        cos_d = (
            tf.sin(pred_pitch) * tf.sin(true_pitch)
            + tf.cos(pred_pitch) * tf.cos(true_pitch)
            * tf.cos(pred_yaw - true_yaw)
        )

        cos_d = tf.clip_by_value(cos_d, -1.0, 1.0)

        error = 1.0 - cos_d

        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class SaveAfterEpoch(keras.callbacks.Callback):
    def __init__(self, start_epoch, filepath):
        super(SaveAfterEpoch, self).__init__()
        self.start_epoch = start_epoch
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            path = self.filepath.format(epoch = epoch) # replace {epoch} placeholder
            self.model.save(path)

### custom layer

class ResidualBlock(layers.Layer):
    def __init__(self, filters, dilation_rate):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation_rate,)
        self.norm1 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation_rate,)
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        residual = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = tf.nn.gelu(y)
        y = self.conv2(y)
        y = self.norm2(y)

        return tf.nn.gelu(y + residual)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model=32, num_heads=2):
        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        self.ffn = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dense(d_model),
        ])

    def call(self, x):
        attn = self.att(x, x)
        x = self.norm1(x + attn)

        ffn = self.ffn(x)
        x = self.norm2(x + ffn)

        return x

class KalmanLayer(layers.Layer):
    def call(self, inputs):

        x_t = inputs[:, -1, :]
        x_prev = inputs[:, -2, :]

        velocity = x_t - x_prev

        return x_t + velocity
