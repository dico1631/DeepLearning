# Loss Function

> 오차를 계산하는 함수
>
> 얼마나 잘못 계산하였는가?

### 1. Mean Squared Error (MSE)

- 적은 error에는 작은 penalty를, 큰 error에는 큰 penalty를 주는 효과가 있음
- 계산이 간단해서 빠름

<img src="https://user-images.githubusercontent.com/58927491/78747891-853aa800-79a5-11ea-87b4-541295eb08b3.png" alt="image" style="zoom: 67%;" />

### 2. Cross Entropy Error (CEE)

- One-hot encoding만 적용
- 같은 상황에도 오차의 값이 더 크게 나오기에 가중치에 update 할 때 더 크게 update 됨
- 적은 error에는 작은 penalty를, 큰 error에는 큰 penalty를 주는 효과가 더 큼

![image-20200408143616659](https://user-images.githubusercontent.com/58927491/78749081-7b667400-79a8-11ea-9202-2ebc5aec52b0.png)

![image](https://user-images.githubusercontent.com/58927491/78748148-3a6d6000-79a6-11ea-9cfe-91f0630c900b.png)



- 단점은 계산이 복잡해서 MSE보다 시간이 오래 걸림 

  -> 오차 역전파에서 update를 크게 할 필요는 없을 때는 MSE

  ​	정확도 높이는 것이 중요한 경우엔 CEE

