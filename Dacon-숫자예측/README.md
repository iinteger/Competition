# 손 글씨 (숫자) 분류 경진대회
#### Dacon Basic 손 글씨 (숫자) 분류 경진대회
<https://dacon.io/competitions/official/235838/overview/description>
</br>
</br>

## 21. 12. 05

* #### 최종 Public Score : 0.939, 리더보드 5위

* #### 최종 Private Score : 0.93725, 리더보드 7위(2%)

  

* MNIST 데이터셋 훈련, 예측하는 대회

* 기존 MNIST 예제와 달리 Train data 5000장, Test data 5000장으로 훈련 데이터 양이 부족하였고, Pseudo labelling을 사용할 수 없었음

* Efficientnet-b0을 10 epoch동안 학습한 결과 **0.817** 달성
</br>

* **해결하고자 하는 문제가 어렵지 않다는 점을 고려해서, 간단한 컨볼루션 넷을 직접 구축해서 훈련함**
</br>
  

* 하나의 컨볼루션 블럭(CB)은 [Conv-Activation-Batchnorm]으로 구성되어 있고, </br>CB-CB-CB-Dropout-CB-CB-CB-Dropout-CB-Dropout-Dense로 구성된 네트워크를 사용

* 동일 네트워크를 15번 이상 앙상블하여 정확도 향상을 노렸고, 이때 정확도 **0.895** 달성

* 훈련 데이터가 부족하므로 aumentation을 적용하였고, 파라미터는 다음과 같음

  * rotation_range=15
  * zoom_range=0.15
  * width_shift_range=0.15
  * height_shift_range=0.1
  * shear_range=0.2
  * fill_mode="nearest"

* **이때 flip은 사용하지 않았고 rotation_range 또한 크지 않은 수치를 사용하였는데, flip 혹은 과도한 rotation은 6-9를 서로 뒤집힌것으로 보이게 할 수 있기 때문**
* 실험적으로 파인튜닝을 하였고, 활성화 함수를 **Swish** 로 사용한 결과, 최종 정확도 **0.939** 달성



* epoch를 늘려서 과도하게 학습하여도 오버피팅이 발생하지 않았는데, 이는 Test data의 분산이 train data를 벗어나지 않는 단순한 예제이기 때문이라고 생각함
* 또한 무조건 큰 모델(efficientnet)이 좋은 결과를 보장하진 않는다는것을 실험적으로 확인함
