import numpy as np
import pandas as pd

import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

from IPython.display import display

DEBUG = False

train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

if DEBUG:
    train = train[:80*1000]


def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()

    #######################################
    # fast area calculation
    df['time_delta'] = df['time_step'].diff()
    df['time_delta'].fillna(0, inplace=True)
    df['time_delta'].mask(df['time_delta'] < 0, 0, inplace=True)
    df['tmp'] = df['time_delta'] * df['u_in']
    df['area_true'] = df.groupby('breath_id')['tmp'].cumsum()

    # u_in_max_dict = df.groupby('breath_id')['u_in'].max().to_dict()
    # df['u_in_max'] = df['breath_id'].map(u_in_max_dict)
    # u_in_min_dict = df.groupby('breath_id')['u_in'].min().to_dict()
    # df['u_in_min'] = df['breath_id'].map(u_in_min_dict)
    u_in_mean_dict = df.groupby('breath_id')['u_in'].mean().to_dict()
    df['u_in_mean'] = df['breath_id'].map(u_in_mean_dict)
    del u_in_mean_dict
    u_in_std_dict = df.groupby('breath_id')['u_in'].std().to_dict()
    df['u_in_std'] = df['breath_id'].map(u_in_std_dict)
    del u_in_std_dict

    # u_in_half is time:0 - time point of u_out:1 rise (almost 1.0s)
    df['tmp'] = df['u_out'] * (-1) + 1  # inversion of u_out
    df['u_in_half'] = df['tmp'] * df['u_in']

    # u_in_half: max, min, mean, std
    u_in_half_max_dict = df.groupby('breath_id')['u_in_half'].max().to_dict()
    df['u_in_half_max'] = df['breath_id'].map(u_in_half_max_dict)
    del u_in_half_max_dict
    u_in_half_min_dict = df.groupby('breath_id')['u_in_half'].min().to_dict()
    df['u_in_half_min'] = df['breath_id'].map(u_in_half_min_dict)
    del u_in_half_min_dict
    u_in_half_mean_dict = df.groupby('breath_id')['u_in_half'].mean().to_dict()
    df['u_in_half_mean'] = df['breath_id'].map(u_in_half_mean_dict)
    del u_in_half_mean_dict
    u_in_half_std_dict = df.groupby('breath_id')['u_in_half'].std().to_dict()
    df['u_in_half_std'] = df['breath_id'].map(u_in_half_std_dict)
    del u_in_half_std_dict

    gc.collect()

    # All entries are first point of each breath_id
    first_df = df.loc[0::80, :]
    # All entries are first point of each breath_id
    last_df = df.loc[79::80, :]

    # The Main mode DataFrame and flag
    main_df = last_df[(last_df['u_in'] > 4.8) & (last_df['u_in'] < 5.1)]
    main_mode_dict = dict(zip(main_df['breath_id'], [1] * len(main_df)))
    df['main_mode'] = df['breath_id'].map(main_mode_dict)
    df['main_mode'].fillna(0, inplace=True)
    del main_df
    del main_mode_dict

    # u_in: first point, last point
    u_in_first_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
    df['u_in_first'] = df['breath_id'].map(u_in_first_dict)
    del u_in_first_dict
    u_in_last_dict = dict(zip(first_df['breath_id'], last_df['u_in']))
    df['u_in_last'] = df['breath_id'].map(u_in_last_dict)
    del u_in_last_dict
    # time(sec) of end point
    time_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))
    df['time_end'] = df['breath_id'].map(time_end_dict)
    del time_end_dict
    del last_df

    # u_out1_timing flag and DataFrame: speed up
    # 高速版 uout1_df 作成
    df['u_out_diff'] = df['u_out'].diff()
    df['u_out_diff'].fillna(0, inplace=True)
    df['u_out_diff'].replace(-1, 0, inplace=True)
    uout1_df = df[df['u_out_diff'] == 1]

    gc.collect()

    # main_uout1 = uout1_df[uout1_df['main_mode']==1]
    # nomain_uout1 = uout1_df[uout1_df['main_mode']==1]

    # Register Area when u_out becomes 1
    uout1_area_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
    df['area_uout1'] = df['breath_id'].map(uout1_area_dict)
    del uout1_area_dict

    # time(sec) when u_out becomes 1
    uout1_dict = dict(zip(uout1_df['breath_id'], uout1_df['time_step']))
    df['time_uout1'] = df['breath_id'].map(uout1_dict)
    del uout1_dict

    # u_in when u_out becomes1
    u_in_uout1_dict = dict(zip(uout1_df['breath_id'], uout1_df['u_in']))
    df['u_in_uout1'] = df['breath_id'].map(u_in_uout1_dict)
    del u_in_uout1_dict

    # Dict that puts 0 at the beginning of the 80row cycle
    first_0_dict = dict(zip(first_df['id'], [0] * len(uout1_df)))

    del first_df
    del uout1_df

    gc.collect()

    # Faster version u_in_diff creation, faster than groupby
    df['u_in_diff'] = df['u_in'].diff()
    df['tmp'] = df['id'].map(first_0_dict)  # put 0, the 80row cycle
    df.iloc[0::80, df.columns.get_loc('u_in_diff')] = df.iloc[0::80, df.columns.get_loc('tmp')]

    # Create u_in vibration
    df['diff_sign'] = np.sign(df['u_in_diff'])
    df['sign_diff'] = df['diff_sign'].diff()
    df['tmp'] = df['id'].map(first_0_dict)  # put 0, the 80row cycle
    df.iloc[0::80, df.columns.get_loc('sign_diff')] = df.iloc[0::80, df.columns.get_loc('tmp')]
    del first_0_dict

    # Count the number of inversions, so take the absolute value and sum
    df['sign_diff'] = abs(df['sign_diff'])
    sign_diff_dict = df.groupby('breath_id')['sign_diff'].sum().to_dict()
    df['diff_vib'] = df['breath_id'].map(sign_diff_dict)

    if 'diff_sign' in df.columns:
        df.drop(['diff_sign', 'sign_diff'], axis=1, inplace=True)
    if 'tmp' in df.columns:
        df.drop(['tmp'], axis=1, inplace=True)

    gc.collect()
    #######################################
    '''
    '''

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    # df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    # df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    # df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    # df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    # df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    # df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    # df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    # df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df['u_in_lag5'] = df.groupby('breath_id')['u_in'].shift(5)
    df['u_in_lag_back5'] = df.groupby('breath_id')['u_in'].shift(-5)
    df['u_in_lag6'] = df.groupby('breath_id')['u_in'].shift(6)
    df['u_in_lag_back6'] = df.groupby('breath_id')['u_in'].shift(-6)
    df['u_in_lag7'] = df.groupby('breath_id')['u_in'].shift(7)
    df['u_in_lag_back7'] = df.groupby('breath_id')['u_in'].shift(-7)
    df = df.fillna(0)

    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    # df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    # df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['cross'] = df['u_in'] * df['u_out']

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)

    # good!
    df["total_u_in_group"] = df.groupby("breath_id")["u_in"].cumsum()
    df.loc[df["u_out"] == 1, "total_u_in_group"] = 0

    # reference by https://www.kaggle.com/swaralipibose/interesting-feature-importance-by-lstm-gradients
    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']

    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / df['count']


    # 빼면 안되는 변수 : area_uout1, breath_id__u_in__max, cross, u_in_uout1
    # 빼야하는 변수 : uout, u_out_diff, main mode
    df.drop(["u_out", "u_out_diff", "main_mode", "one", "count"], axis=1, inplace=True)
    df = pd.get_dummies(df)
    return df


train = add_features(train)
test = add_features(test)

print(train.head())

targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)

RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


class WarmupExponentialDecay(Callback):
    def __init__(self, lr_base=0.0002, lr_min=0.0, decay=0, warmup_epochs=0):
        self.num_passed_batchs = 0  # 一个计数器
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base  # learning_rate_base
        self.lr_min = lr_min  # 最小的起始学习率,此代码尚未实现
        self.decay = decay  # 指数衰减率
        self.steps_per_epoch = 0  # 也是一个计数器

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch == 0:
            # 防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr * ((1 - self.decay) ** (
                                    self.num_passed_batchs - self.steps_per_epoch * self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
        # 用来输出学习率的,可以删除
        print("learning_rate:", K.get_value(self.model.optimizer.lr))


EPOCH = 300
BATCH_SIZE = 256
NUM_FOLDS = 3

TPU = False

if TPU:
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

    ## instantiate a distribution strategy
    xpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # GET GPU STRATEGY
    # xpu_strategy = tf.distribute.get_strategy()
    xpu_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with xpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-' * 15, '>', f'Fold {fold + 1}', '<', '-' * 15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]

        model = keras.models.Sequential([
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(768, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            #             keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation='selu'),
            #             keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        model.compile(optimizer=optimizer, loss="mae")

        #         scheduler = ExponentialDecay(1e-3, 40*((len(train)*0.8)/BATCH_SIZE), 1e-5)
        #         lr = LearningRateScheduler(scheduler, verbose=1)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        #         lr = WarmupExponentialDecay(lr_base=1e-3, decay=1e-5, warmup_epochs=30)
        es = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min", restore_best_weights=True)

        checkpoint_filepath = f"folds{fold}.hdf5"
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )

        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE,
                  callbacks=[lr, es, sv])
        # model.save(f'Fold{fold+1} RNN Weights')
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

submission["pressure"] = sum(test_preds)/NUM_FOLDS
submission.to_csv('submission.csv', index=False)

submission["pressure"] = np.median(test_preds, axis=0)
submission.to_csv('submission_median.csv', index=False)