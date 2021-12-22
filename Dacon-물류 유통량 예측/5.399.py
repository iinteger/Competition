import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna


seed = 42
np.random.seed(seed)

train = pd.read_csv('data/train_df.csv', encoding='cp949')
test = pd.read_csv('data/test_df.csv', encoding='cp949')
submission = pd.read_csv('data/sample_submission.csv')

train["SEND_SPG_INNB"] = (train["SEND_SPG_INNB"] // 10000000000) * 10000000000
train["REC_SPG_INNB"] = (train["REC_SPG_INNB"] // 10000000000) * 10000000000
test["SEND_SPG_INNB"] = (test["SEND_SPG_INNB"] // 10000000000) * 10000000000
test["REC_SPG_INNB"] = (test["REC_SPG_INNB"] // 10000000000) * 10000000000

train_X = train.drop('INVC_CONT',axis = 1)
train_X = train_X.drop(['index'], axis=1)
train_y = train['INVC_CONT']

test_X = test.drop(['index'], axis=1)

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=seed)

# optuna

EARLY_STOPPING_ROUND = 100
def objective(trial):
    param = {}
    param['learning_rate'] = trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.15])
    param['depth'] = trial.suggest_int('depth', 14, 16)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 5.0, 6.0, 0.5)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [3, 4, 5])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 10000
    param['use_best_model'] = True
    param['eval_metric'] = 'RMSE'
    param['od_type'] = 'iter'
    param['od_wait'] = 20
    param['random_state'] = seed
    param['logging_level'] = 'Silent'

    regressor = CatBoostRegressor(**param)

    regressor.fit(train_X.copy(), train_y.copy(),
                  eval_set=[(valid_X.copy(), valid_y.copy())],
                  early_stopping_rounds=EARLY_STOPPING_ROUND,
                  cat_features=["DL_GD_LCLS_NM", "DL_GD_MCLS_NM"])
    loss = mean_squared_error(valid_y, regressor.predict(valid_X.copy()), squared=False)
    return loss

# study = optuna.create_study(study_name=f'catboost-seed{seed}')
# study.optimize(objective, n_trials=10000, timeout=24000)
# print(study.best_value)
# print(study.best_params)


model1 = CatBoostRegressor(learning_rate=0.1, depth=15, l2_leaf_reg=5.5, min_child_samples=4,iterations=10000 , grow_policy="Depthwise", eval_metric="RMSE", random_state=seed)
model1.fit(train_X, train_y, cat_features=["DL_GD_LCLS_NM", "DL_GD_MCLS_NM"], eval_set=[(valid_X, valid_y)], early_stopping_rounds=100)

pred = model1.predict(test_X)

submission['INVC_CONT'] = np.around(pred)
print(submission.INVC_CONT.unique())

# 음수 및 0이 나오는 경우를 처리해야함 -> 최빈값으로
for _, row in submission.iterrows():
    if row["INVC_CONT"] <= 0:
        LCLS = test[test["index"] == row["index"]]["DL_GD_LCLS_NM"].item()
        MCLS = test[test["index"] == row["index"]]["DL_GD_MCLS_NM"].item()

        train_LCLS = train[train['DL_GD_LCLS_NM'] == LCLS]
        train_LCLS_MCLS = train_LCLS[train_LCLS['DL_GD_MCLS_NM'] == MCLS]
        mode_value = train_LCLS_MCLS["INVC_CONT"].mode()

        submission.loc[submission["index"] == row["index"], "INVC_CONT"] = mode_value.item()

print(submission.INVC_CONT.unique())
submission.to_csv('baseline.csv',index = False)