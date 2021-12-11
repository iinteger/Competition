# 생육 기간 예측 경진대회
#### Dacon 생육 기간 예측 경진대회
<https://dacon.io/competitions/official/235851/overview/description>
</br>
</br>

## ~ 21. 12. 11

* **다른 성장 일자를 가진 식물 이미지 한쌍의 생육 일자 차이를 추론하는 대회**

  * ex) x1 : 청경채 22일차, x2 : 청경채 9일차, y : 13

  

* Regnetx_002를 사용한 baseline 결과는 6.824(RMSE)

* 식물의 색깔이 대부분 초록색인데서 착안해서 Green 채널만 정규화 하지 않고 raw data 그대로 학습하였는데 정확도는 떨어짐

* 간단한 augmentation을 적용하고, valid set를 나누어 Efficientnet 등 다양한 모델을 학습중. 이후 우수한 결과물들로 앙상블할 예정
