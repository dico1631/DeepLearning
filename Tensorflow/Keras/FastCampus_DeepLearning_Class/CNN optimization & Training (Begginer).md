# CNN optimization & Training (Begginer)

> 만든 모델을 사용하고, 더 나은 모델이 되도록 오차를 계산, 파라미터를 최적화한다.

![image](https://user-images.githubusercontent.com/58927491/77527797-2d457100-6ed0-11ea-96f0-8976893aa101.png)

```python
# data 가져오고, modeling 하기

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

inputs = layers.Input((28, 28, 1))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
```



### 1. Optimization, Loss, Iogit(fit) 

> 모델을 fit 하기전에 미리 사용될 loss 함수와 optimization 방법, 모델의 성능이 향상되는 지를 평가할 방법인 metrics를 설정한 다음에 fit을 한다.

### 1) Optimization

> 파라미터(가중치, w)를 변경할 때 어떤 방식으로 계산해서 update를 할 지를 정하는 것

```python
optm = tf.keras.optimizers.Adam() #opt 방식으로 adam이라는 걸 채택함
```

### 2) Loss 함수

> 예상과 실제의 차이, 즉 오차를 구하는 방법을 정하는 것
>
> category인지 Binary인지 / 문자여서 숫자로 변환이 필요한지 아닌지 등을 설정한다.

- Categorical vs Binary

   : 분류 카테고리가 2개인지(Binary) 3개 이상인지(Categorical) 여부

```python
loss = 'binary_crossentropy' # 2개
loss = 'categorical_crossentropy' # 3개 이상
```

- sparse_categorical_crossentropy vs categorical_crossentropy

  : one-hot encoding이 필요한지 여부

  문자여서 숫자로 변경하는 과정을 one_hot_encoding이라고 한다.

  퍼셉트론은 수치를 계산해서 class를 내보내는 것이기에 class 분류도 숫자로 되어야 해서 진행

  ex) 이 문자가 0,1,2 중에 어떤 숫자인가 > (1,0,0) = 0이다

  ​	 : 0번째 자리가 0, 1번째 자리가 1, 2번째 자리가 2이기에 encoding 필요 없음

  ex) 이 문자가 a, b, c 중에 어떤 문자인가 > (0,1,0) = b다? 

  ​	: encoding을 안하면 어떤 문자가 어떤 자리수인지 알 수 없기에 장담할 수 없음. encoding 필요

  ​

```python
tf.keras.losses.categorical_crossentropy
tf.keras.losses.binary_crossentropy
```

### 3) Metrics

> 모델을 평가하는 방법

```python
metrics = [tf.keras.metrics.Accuracy()] # test 전체 중에 맞춘 비율
tf.keras.metrics.Precision() # True라고 분류한 것 중에서 실제 True인 것의 비율
tf.keras.metrics.Recall() # 실제 True인 것 중에서 모델이 True라고 예측한 비율
```

관련 내용 링크 : <https://sumniya.tistory.com/26>



### 4) compile

> 위에서 정한 optimizer, loss, metrics를 미리 만들어 둔 모델에 적용시키는 코드

```python
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.Accuracy()])
```



### 5) Iogit(fit)

1. 학습용 Hyperparameter 설정

   - num_epochs : dataset 1회독 = 1epchs
   - batch_size : 메모리 한계가 있기 때문에 정해주는 1번에 처리하는 단위

2. model.fit

   ```python
   num_epochs = 1
   batch_size = 32

   model.fit(train_x, train_y, 
             batch_size=batch_size, 
             # shuffle을 안하면 차례대로 값이 들어간다.
             # 그렇게 되면 초기 값의 영향력은 미비해지고,
             # 마지막에 들어오는 값의 영향력이 너무 커지게 된다.
             # 또한 shuffle을 통해 과적합을 방지할 수 있다.
             shuffle=True, 
             epochs=num_epochs) 
   ```

   ​



#### * CNN 코딩 순서 정리

data 가져와서 전처리 > model 만들기(feature extraction과 classification) > model.compile > model.fit > 학습 과정(History) 결과 확인