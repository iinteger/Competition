from typing import Dict
import numpy as np
import pandas as pd
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

start = time.time()

DATA_PATH = "data/"
SEED = 42
np.random.seed(SEED)

def add_code(
        df: pd.DataFrame,
        d_code: Dict[int, Dict[str, int]],
        h_code: Dict[int, Dict[str, int]],
        l_code: Dict[int, Dict[str, int]],
) -> pd.DataFrame:
    # Copy input data
    df = df.copy()

    # D Code
    df['person_prefer_d_1_n'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_1_s'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_1_m'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_1_l'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['person_prefer_d_2_n'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_2_s'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_2_m'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_2_l'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['person_prefer_d_3_n'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_3_s'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_3_m'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_3_l'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['contents_attribute_d_n'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['contents_attribute_d_s'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['contents_attribute_d_m'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['contents_attribute_d_l'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    # H Code
    df['person_prefer_h_1_l'] = df['person_prefer_h_1'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_1_m'] = df['person_prefer_h_1'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    df['person_prefer_h_2_l'] = df['person_prefer_h_2'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_2_m'] = df['person_prefer_h_2'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    df['person_prefer_h_3_l'] = df['person_prefer_h_3'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_3_m'] = df['person_prefer_h_3'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    df['contents_attribute_h_l'] = df['contents_attribute_h'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['contents_attribute_h_m'] = df['contents_attribute_h'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    # L Code
    df['contents_attribute_l_n'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 세분류코드'])
    df['contents_attribute_l_s'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 소분류코드'])
    df['contents_attribute_l_m'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 중분류코드'])
    df['contents_attribute_l_l'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 대분류코드'])

    return df


d_code = pd.read_csv('data/속성_D_코드.csv', index_col=0).T.to_dict()
h_code = pd.read_csv('data/속성_H_코드.csv', index_col=0).T.to_dict()
l_code = pd.read_csv('data/속성_L_코드.csv', index_col=0).T.to_dict()

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_train = add_code(df_train, d_code, h_code, l_code)
df_test = add_code(df_test, d_code, h_code, l_code)

drop_cols = ["id", "person_prefer_f", "person_prefer_g", "contents_open_dt", "contents_rn"]

df_train.drop(drop_cols, axis=1, inplace=True)
df_test.drop(drop_cols, axis=1, inplace=True)

X_train = df_train.drop("target", axis=1)
y_train = df_train["target"]
X_test = df_test

cat_features = X_train.columns[X_train.nunique() > 2].tolist()
numeric_features = ["person_attribute_a_1", "person_attribute_b", "person_prefer_e", "contents_attribute_e"]
cat_features = [i for i in cat_features if i not in numeric_features]

is_holdout = False
trials = 100
iterations = 10000
patience = 50
test_size = 0.20

scores = []
models = []

for trial in range(trials):
    print("=" * 50)
    print("trial", trial)
    preds = []
    model = CatBoostClassifier(iterations=iterations, random_state=trial, task_type="GPU", eval_metric="F1", cat_features=cat_features, one_hot_max_size=10, use_best_model=True)
    x_train_trial, x_valid_trial, y_train_trial, y_valid_trial = train_test_split(X_train, y_train, test_size=test_size, random_state=trial, stratify=y_train)

    model.fit(x_train_trial, y_train_trial,
              eval_set=[(x_valid_trial, y_valid_trial)],
              early_stopping_rounds=patience,
              verbose=0,
              )

    models.append(model)

threshold = 0.39

pred_list = []
for trial in range(trials):
    pred = models[trial].predict_proba(X_test)[:, 1]
    pred_list.append(pred)

pred = np.mean(pred_list, axis=0)
pred = np.where(pred >= threshold, 1, 0)

sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')
sample_submission['target'] = pred

sample_submission.to_csv("prediction.csv", index=False)

print("time :", time.time() - start)