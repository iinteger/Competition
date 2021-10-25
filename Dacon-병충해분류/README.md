# 작물 병해 분류
#### Dacon 작물 병해 분류 AI 경진대회 기록
<https://dacon.io/competitions/official/235842/overview/description>
</br>
</br>

## 21. 10. 24

* #### 최종 Score : 0.98334, 리더보드 14위 기록. 상위 8%이며 15위까지 수상

  

* 대회 종료 4일 전에 참여하여 많은 시도는 해보지 못함

* Baseline으로 제공된 Resnet-50과 Efficientnet-b7을 20:80으로 앙상블하여 사용

* training data가 250장으로 굉장히 적었음

* 다양한 augmentation을 적용해보진 못했으나, blur처럼 색온도를 변화시키는 전처리는 성능을 악화시켰음. 병해 판단에 색상 정보가 많이 기여했던것으로 예상됨. 따라서 random flip과 crop만 사용

* batchsize, epoch등의 하이퍼 파라미터는 실험적으로 최적이라고 생각되는 값을 사용

</br>

  

* k-fold를 사용하지 못한것이 아쉬움

* 훈련 데이터가 적었고 class imbalance가 심각하였으며 평가 metrc이 f1 score였음. 이를 대처하지 못했는데 상위 수상자들은 weighted loss function이나 under sampling등의 방법으로 해결하고자 하였음. 또한 공통적으로 test data를 pseudo labeling하여 사용하였는데, data leakage가 아닌지 궁금함

* baseline score : 0.90374 -> 최종 score : 0.98766으로 상승

