# 사물 이미지 분류 경진대회

#### Dacon Basic 사물 이미지 분류 경진대회

[사물 이미지 분류 경진대회 - DACON](https://dacon.io/competitions/official/235874/overview/description)

<div>
</br>
</div>

## 2022. 03. 04

- #### Public Score : 0.919, 4/235 등, 상위 1.7%
- #### Private Score : 0.914, 4/235 등, 상위 1.7%
</br>

* Cifar-10 데이터를 분류하는 Basic 대회

* Train data , Valid data를 8:2로 나누어 사용

* 다음 Augmentation 사용
  
  * transforms.Resize((img_size, img_size))
  
  * transforms.RandomHorizontalFlip()
  
  * transforms.RandomRotation(15)
  
  * transforms.ToTensor()
  
  * transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

* Optimizer는 Adam과 AdamW를 혼용, Learning Rate는 0.003 사용

* CosineAnnealingLR 스케줄러 사용

* 기존 데이터 size가 32x32로 작기 때문에 112, 224 등으로 확대하여 사용함. 이미지에 왜곡이 발생하더라도 학습에 사용한 EfficientNet의 Input Size에 근접한 값을 사용하는것이 좋다고 판단

* 다음 여섯가지 모델을 Soft Voting하여 결과를 예측
  
  * Resnet-50
  
  * Densenet-201
  
  * EfficientNet-b0
  
  * EfficientNet-b3
  
  * EfficientNet-b4
  
  * EfficientNet-b7

* TTA를 5번 한 평균값을 사용했을때보다 TTA를 사용하지 않았을 때 결과가 더 좋게 나왔음
  * Augmentation이 모델의 Robustness는 향상시키지만 목표 데이터셋의 Top-Accuracy에는 크게 영향을 끼지 않는 것으로 예상됨
