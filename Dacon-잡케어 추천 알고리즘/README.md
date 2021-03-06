# 잡케어 추천 알고리즘 경진대회

#### Dacon 잡케어 추천 알고리즘 경진대회

<https://dacon.io/competitions/official/235863/overview/description>
</br>
</br>

## ~ 22. 01. 09

* #### 구직자의 다양한 속성을 사용하여 컨텐츠 사용 여부를 예측

* 공유된 코드 중 Catboost 기반의 코드를 baseline으로 사용

* Optuna로 찾은 하이퍼파라미터를 사용해도 default parameter보다 public score가 낮게 나옴. 기존 점수와의 차이도 유의미하게 나타나지는 않음

* 코드공유에서는 threshold를 0.4로 낮추는 방법으로 public score를 크게 높임. 그러나 target data가 imbalance하지는 않기 때문에 오히려 overfitting을 발생시키는것이 아닐까 의심됨

</br>

## 22. 01. 10 ~ 22. 01. 14

* 기존 코드는 k-fold를 사용하므로 폴드 수를 늘리면 early stopping에 사용하는 validation data가 줄어들기 때문에 overfitting의 위험이 있음. 보다 많은 모델을 사용하기 위해 k-fold 방식에서, 매 훈련마다 20%의 validation data를 random sampling하는 방법으로 바꿈. 최적의 validation ratio는 실험적으로 테스트중
* public score가 전체 테스트 데이터의 33%라는것에서 착안하여, public score를 높이는것이 private socre의 향상에도 도움이 될 것이라고 예상됨. 따라서 threshold를 valid score가 가장 높게 나오는 0.39로 설정. (validation ratio에 따라서 바뀔 수 있음)

<div>
</br>
</div>

## ~ 2022. 01. 29

* id, contents_open_df, contents_rn 컬럼은 유의미하지 않고, person_prefer_f와 person_prefer_g는 1개의 값으로만 이루어진 컬럼이므로 제거

* person_attribute_a_1, person_attribute_b, person_attribute_e, contents__attribute_e 컬럼은 값 차이에 의미가 있는 컬럼이기 때문에 cat_feature에서 제외

* 보다 많은 시행을 하기 위해 K-fold 방법에서 random sampling 방법으로 변경. 100번의 훈련-추론을 시행하며 각 시행마다 8:2 random sampling을 진행. 매 스플릿에는 stratify가 적용되었으며, 다양성을 주기 위해 random state를 달리 주었음

* 하이퍼 파라미터 튜닝은 따로 하지 않았고, one_hot_max_size 변수만 10으로 변경함

<div>
</br>
</div>

## 2022. 01. 30

* #### Public Score : 0.70109, 111등, 15.2%

* #### Private Score :  0.70197, 104등, 14.2%








