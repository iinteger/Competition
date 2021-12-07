import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor

trainset = pd.read_csv('./data/train.csv')
testset = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')


def split_date(df):
    df['year'] = pd.to_datetime(df['date_time']).dt.year
    df['month'] = pd.to_datetime(df['date_time']).dt.month
    df['day'] = pd.to_datetime(df['date_time']).dt.day
    df['week'] = pd.to_datetime(df['date_time']).dt.week
    df['weekday'] = pd.to_datetime(df['date_time']).dt.weekday
    df["one"] = 1
    df["day_cumsum"] = df["one"].groupby(df["year"]).cumsum()

    return df.drop(columns=['date_time',
                            'one',
                            'wind_direction',
                            'Precipitation_Probability',
                            'month',
                            'week'])


traindf = split_date(trainset)
testdf = split_date(testset)

keys = traindf.drop(columns=['number_of_rentals']).keys()

keydict = dict()
for i, k in enumerate(keys):
    keydict[i]=k

X = np.array(traindf.drop(columns=['number_of_rentals'])).astype(float)
y = np.array(traindf['number_of_rentals']).astype(float)
X_test = np.array(testdf).astype(float)


def nmae(true, pred):
    return np.mean(np.abs(pred-true)/true)


def select_features(X, cols):
    _X = list()
    for c in cols:
        _X.append(np.expand_dims(X[:, c],-1))
    return np.concatenate(_X, axis=1)


min_loss = 1e10
min_cols = None

cols = np.arange(len(keys))
_X = select_features(X, cols)

model = XGBRegressor(tree_method='gpu_hist')
model.fit(_X, y)

_X_test = select_features(X_test, cols)
y_pred = model.predict(_X_test)
y_pred = y_pred.astype(int)

submission['number_of_rentals'] = np.around(y_pred)
submission.to_csv('submission.csv', index=False)