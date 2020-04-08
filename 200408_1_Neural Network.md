# Neural Network



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

- hidden layer

  - Linear Function

    ![image](https://user-images.githubusercontent.com/58927491/78734896-6677e980-7984-11ea-9d09-47669ce886f6.png)

  - Activation Function

    : 임계점(Threshold)를 넘으면 값을 전달하고 아니면 없애거나, 혹은 원하는 형식으로 값을 변환시키는 필터 역할

    

    1) step Function

    - 0인 값은 사라지고 0 이상인 값만 살아남는다.

    
  
    2) sigmoid Function

    - 0 < F < 1 범위의 숫자로 값을 변환 : 2진 분류에서 class별 확률을 위해 나온다.
  - y =1 / (1 + e^(-x))    e는 자연상수 = 2.71828…
    - 단점 : 깊이가 깊어지면 vanishing gradient problem 발생
  - 은닉층 중간에는 안 쓰이고 마지막에 분류를 위해서만 사용됨
  
  
  
    3) tanh(a) = (exp(2a) - 1) / (exp(2a) + 1) = sinh(x) / cosh(x)
  
    - sigmoid와 비슷한데 0과 1을 지난다. 더 classification 가능성이 다양해진다. 
  
    
  
    4) Relu
  
    - 0 이하 = 0 / 0 초과 = 그대로 변환 ( max(0, x) )
    - vanishing gradient problem 해결
  
    
  
    5) softmax
  
    - 입력 받은 값을 0~1사이의 출력 값으로 정규화(총합은 1) 
  
      = 각 class가 될 확률 = 다중 분류 문제
  
      <img src="https://user-images.githubusercontent.com/58927491/78739526-6f21ed00-798f-11ea-9103-0d7b79d1b5b3.png" alt="image" style="zoom: 20%;" />
  
    - 결과 값을 One hot encoder의 입력으로 사용하면, 가장 큰 값만 True
    - 은닉층 중간에는 안 쓰이고 마지막에 분류를 위해서만 사용됨



- fully connected

  : 모든 노드들이 연결되어 있는 상태, Feed forward와 Back propagation 둘 다 fully connected 하다.

  - 오차가 생기면 그건 모든 w들이 문제여서 생긴 것이기에 모든 w에게 해당 오차의 변동을 나눠줘야한다. > Back propagation

  ![image](https://user-images.githubusercontent.com/58927491/78744197-e78eab00-799b-11ea-83cc-059bf6f79a87.png)

  