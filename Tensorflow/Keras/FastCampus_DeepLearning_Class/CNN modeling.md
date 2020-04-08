# CNN model 만들기

> 사진 / 영상처리를 위한 인공지능 기술
>
> 사진을 보고 어떤 것인지 구분하는 기술



- CNN은 2가지 단계로 나눠진다.
  - feature extraction : 사진의 특징을 뽑아내는 작업. 두드러지는 특징은 남기고, 사소한 data는 걸러내면서 data의 양을 분석할 수 있도록 줄인다.
  - classification : feature extraction을 통해 걸러진 데이터를 input으로 넣고, 은닉층을 거치면서 어떤 사진인지 classification을 하는 작업



### 1. feature extraction: 특징 추출

> 사진의 특징적인 부분만을 추출하는 것
>
> convolution과 pooling을 반복해서 시행한다.

#### 1) convolution: 합성곱

​	사진과 filter를 연산해서 사진의 선 중에 두드러지는 선만을 남긴다.

![image](https://user-images.githubusercontent.com/58927491/77523459-1e0ef500-6ec9-11ea-9673-997363a5387d.png)

```python
tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='VALID', activation='relu')

# 위 코드와 같은 코드
tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1, 1), padding='VALID', activation='relu')
```

filters: layer에서 나갈 때 몇 개의 filter를 만들 것인지 (a.k.a weights, filters, channels)  

kernel_size: filter(Weight)의 사이즈  

strides: 몇 개의 pixel을 skip 하면서 훑어지나갈 것인지 (사이즈에도 영향을 줌)  

padding: zero padding을 만들 것인지. VALID는 Padding이 없고, SAME은 Padding이 있음 (사이즈에도 영향을 줌)  

activation: Activation Function을 만들것인지. 당장 설정 안해도 Layer층을 따로 만들 수 있음



#### 2) pooling : 정보 축소

![image](https://user-images.githubusercontent.com/58927491/77524348-88746500-6eca-11ea-8a0e-e3eee947dc5f.png)



```python
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
output = layer(output)
```

pool_size: 결과로 나오는 output feature map의 크기

strides: 축소할 단위의 크기

padding: convolution의 padding과 같음



### 2. classification: 분류

> 퍼셉트론 시작
>
> 사진에 나온 것이 무엇인지 분류를 시작
>
> ex) 개인지 고양이인지 구분하는 것



#### 1) Fully Connected 

>  classification을 위한 기본적인 구조, 모든 노드가 다 연결된 상태를 fully connected라고 한다.

![image](https://user-images.githubusercontent.com/58927491/77525494-78f61b80-6ecc-11ea-9ad3-008e2f9affed.png)

#### 2) Flatten

>  feature extraction에서 convolution과 pooling을 통해 만든 pooled feature map을 input 노드로 넣기 위해서 한줄로 만드는 것

![image](https://user-images.githubusercontent.com/58927491/77525564-97f4ad80-6ecc-11ea-9865-656f28cb756e.png)

```python
layer = tf.keras.layers.Flatten()
output = layer(output)
```



#### 3) Dense

>은닉층의 깊이
>
>각 은닉층에서는 w와 b를 통해 계산 후, activation function을 통해 중요한 요소만 거른다.

![image](https://user-images.githubusercontent.com/58927491/77526317-bc04be80-6ecd-11ea-8155-e0afa94ca3bc.png)

```python
layer = tf.keras.layers.Dense(32, activation='relu')
output = layer(output)
```



#### 4) DropOut

> fully connected 된 상태로 계산하면 시간이 오래 걸리기 때문에 속도 향상을 위해 중요하지 않다고 생각되는 연결선을 자동으로 잘라내는 것

![image](https://user-images.githubusercontent.com/58927491/77526555-2584cd00-6ece-11ea-9082-e16f2c62e561.png)

```python
# 70의 노드만 남기고, 30프로의 노드는 삭제시킨다.
layer = tf.keras.layers.Dropout(0.7)
output = layer(output)
```





## CNN model 만들기(전체 과정)

> 위에서 배운 내용을 총집합한 코드

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

input_shape = (28, 28, 1)
num_classes = 10
inputs = layers.Input(input_shape)

# feature extraction 과정: conv와 pooling의 반복
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

# classification 과정: Flatten하고 Dense를 쌓으면서 model을 만듬
# Dropout을 해서 속도를 높임
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
```

