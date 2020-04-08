# Machine Learning



## 1. 순서

- 학습 단계 ( 데이터 수집 > 피쳐 정의 > 가설 정의 > 비용 함수 정의 > 학습) > 예측 단계
- 학습단계 : Input > H ( Linear F > Activation F ) > Output > Loss F를 통해 parameter 변경, Accuracy에 따라 종료 여부 결정
- 예측단계 : Input > Model > Output
- 모델 만드는 것이 끝나면 예측 시작
- total data = training data + test data (일반적으로 7:3 / 8:2)

- Input : Instatence 1개가 가진 feature 들 > instatence 개수만큼 돌리면 됨



## 2. Classification

- Learning rate

  - 오차가 생겼을 때, 얼마나 update 해야 할 까? 

    1. y_pred = w * x
    2. 오차 E : y - y_pred
    3. y = (w + w') * x

    -> E = (w + w') * x - (w * x) = w' * x

    -> w' = E / x

    그런데 이 식대로 하면 data에 따라 너무 많이 변해서 치우치게 된다!

    그러면 100% 다 반영하지 말고 조금만 반영하면 되지 않을까?

  - L : Learning rate, 0~1 사이의 값으로 새로운 값의 몇 %를 반영해 update를 할 지 결정한다.

    -> w' = L (E / x)

  - Learning rate는 data에 따라 다름. 절대적 수치 X



## 3. 신경망 네트워크

