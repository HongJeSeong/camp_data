# BigData AI Camp 홍대

---

# 분류 : 와인, COVID 증상 판별, Text 데이터 다루기(문장 긍/부정 판별)
# 예측 : COVID 확진자 수 예측(한국)

---


# Keras
딥러닝 개발에 필요한 Tensorflow의 `다양한 모델과 기능을 API로 쉽게` 가져올 수 있다. 

내부적으로 복잡한 구조를 알 필요 없이*(알면 좋음)* 직관적인 API를 가져와  
짧은 몇 줄의 코드로 다양한 모델을 쉽게 개발을 할 수 있는 장점이 있다.
> Tensorflow만으로 개발을 할 경우 더 세부적인 코딩으로 조작이 가능  
> `Tensorflow`만으로 개발 하는 것을 차량의 수동 기어라면 (성능에 최적화 가능, but 불편) 
> `케라스`는 차량의 오토메틱 기어라 비유 할 수 있음 (간편하다) 

---

## Keras로 모델 작성하기

# 딥러닝 모델 작성 과정
## 1. Data Set 생성하기
**머신러닝/딥러닝 수행에 있어서 가장 중요함(비정형된 빅데이터 수집, 전처리 등)  
딥러닝을 잘 수행하는 데에 중요도 비중이 데이터가 80%, 모델 구성이 20% 라고 할 정도로 중요**
- 데이터를 수집하고 분석
- 데이터 전처리 진행
  - 결측치 제거, 정규화, 표준화, 인코딩 과정 등을 수행
- 데이터`[Training Set, Validation Set, Test Set]`로 구성
  - Training Set -> 훈련용 데이터
  - Validation Set -> 훈련된 모델을 검증하는 용도의 데이터
  - Test Set -> 훈련, 검증용에 사용되지 않은 전혀 새로운 데이터로써 테스트 용도의 데이터  

![dataset](https://raw.githubusercontent.com/HongJeSeong/camp_data/main/img/dataset.PNG)  

>100만개의 데이터 경우 Test Set을 10%로 해도 10만개의 데이터로 충분하기 때문 

---

## 2. 모델 구성하기
add() 함수 사용

```
python code example
model = Sequential()
model.add(...)
model.add(...)
model.add(...)
model.add(...)
```
![model](https://raw.githubusercontent.com/HongJeSeong/camp_data/main/img/model.PNG)
**각 layer에서는 input 값과 W(Weight), b(Bias)를 곱, 합연산을 통해 a=WX+b를 계산하고**  

**마지막에 활성화 함수를 거쳐 h(a)를 출력**  

- 활성화 함수 : 데이터를 입력받아 이를 적절한 처리를 하여 출력해주는 함수. 이를 통해 출력된 신호가 다음 단계에서 `활성화 되는지를 결정`  

- 활성화 함수 종류: `sigmoid, softmax, relu `등  

---

## 3. 모델 학습과정 설정하기  
compile() 함수를 사용
```
python code example
model.compile(loss='mean_squared_error', optimizer='adam')
```
`손실 함수`와 `최적화 방법`에 대한 설정 
### 손실 함수(Loss Function)
- 실제값과 예측값의 차이(loss, cost)를 수치화해주는 함수
- `오차가 작을 수록` 좋은 모델
- `손실 함수의 값을 최소화 하는 W(weight), b(Bias)`를 찾아가는것이 학습모델의 목표!(분류, 예측, 인식 등을 잘 하는 모델)
- `RMSE`, `MSE` 등  

### 최적화 함수(Optimizer), 목적 함수
손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것을 최적화 함수를 통해 정한다

![model](https://raw.githubusercontent.com/HongJeSeong/camp_data/main/img/optimizer.PNG)
- 여러 종류가 있고, 각각 장단점이 있음
- 학습 데이터와 구성한 모델에 따라 적용하는 것이 다르지만, 잘 모르겠다면 일단 무난하고 성능도 좋은 `Adam` 옵티마이저를 사용한다!
- `Adam` 옵티마이저가 무조건적으로 좋다는 말이 아님

---

## 4. 모델 학습시키기
fit() 함수를 사용  

데이터 & 구성한 신경망 모델을 학습
```
python code example
model.fit(x_train, y_train, epochs=50,batch_size=128, validation_data=(x_valid, y_valid))
```
모델 학습 시 훈련셋, 검증셋의 손실 및 정확도를 측정
- epochs -> 반복 훈련 횟수
- batch_size -> 1번 훈련할 때 입력하는 데이터의 크기  

>1번 훈련에 10개의 데이터가 필요할 때 batch_size를 2로 하면 5번으로 나누어 데이터 입력  

`컴퓨터의 메모리 문제`를 위한 것으로 배치 사이즈가 `작을 수록` 메모리 자원 사용이 적지만, 학습에 걸리는 시간이 증가  

배치 사이즈가 `크게 된다면?` 메모리에 입력되는 데이터가 커지지만 빠른 학습 가능

---

## 5. 모델 평가 및 예측
- 평가 : evaluate() 함수
- 예측 : predict() 함수 

---

# 맛보기로 제곱 계산 방법 학습 시키기
![learning](https://raw.githubusercontent.com/HongJeSeong/camp_data/main/img/learning.PNG)


```
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 구성하기
x = np.array([-2, -1, 1, 2, 3, 4, 5, 8]) 
y = np.array([4, 1, 1, 4, 9, 16, 25, 64]) #정답데이터

# 모델 구성하기
model = Sequential()
model.add(Dense(32, input_dim=1, activation="relu")) # 32개의 노드 수, 1개의 입력 데이터
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")

# 모델 학습하기
model.fit(x,y,epochs=10000,verbose=0) # verbose(1:과정 출력x,2:과정 출력o)

# 예측 해보기
result = model.predict(x)  # 기대하는 예측 값 -> 4, 1, 1, 4, 9, 16, 25,64
print(result)
```

```
# 학습에 사용되지 않은 새로운 데이터(Test Data)로 해보기
result = model.predict([0,6,7])  # 기대하는 예측 값 -> 0, 36, 49
print(result)
```
