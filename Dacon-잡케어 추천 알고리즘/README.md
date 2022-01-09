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

  

