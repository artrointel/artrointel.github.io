---
layout: post
title: Neural Network for mnist
date: 2017-09-12 13:32:20 +0300
description: Experimental c++ with openMP for mnist example
img: mnist-result.jpg
tags: [mnist, neural network, nn, cnn]
language: kr
---
# (공책 내용 옮기는중..)

## Neural Network Theories focused on Mathematics

먼저 Universal Approximation Theorem을 알아볼 필요가 있다.

----------------------------------------------------------------------------------------------------------------------------------

### 1. Universal Approximation Theorem [[WIKI](https://en.wikipedia.org/wiki/Universal_approximation_theorem), [Reference](http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html)]

임의의 함수 $ F(x) $가 연속한 실수공간에 존재하는 함수라 할 때, 아래와 같이 이에 근사하는 연속함수 $ f(x) $가 존재한다.  
  
$$ F(x) \simeq \sum_{i=1}^N v_i \Phi(w_i^Tx + b_i) = f(x) \text{ where } i \in 1, ..., N $$
  
이 때 $ \Phi() $는 비상수인 단조증가연속함수이며, 충분히 작은 $ \epsilon > 0 $에 대하여 $ |F(x) -  f(x)| > \epsilon $을 만족시키는  
자연수 $ N $ 과 실수 $ v_i, b_i $ 및 벡터 $ w_i $가 존재한다.  


(특히, $ \Phi() $가 sigmoid 함수인 경우 [시벤코 정리](https://ko.wikipedia.org/wiki/%EC%8B%9C%EB%B2%A4%EC%BD%94_%EC%A0%95%EB%A6%AC)라 한다.)

정리에 의해 임의의 함수 $ F(x) $는 Single Perceptron만으로 $ F(x) $에 근사하는 임의의 함수 $ f(x) $를 만들어낼 수 있다는 것을 의미한다.  

----------------------------------------------------------------------------------------------------------------------------------

### 2. Single Perceptron  
  * Notation  
  아래와 같이 d번째 Layer에 존재하는 임의의 index i를 가지는 $ x $를 다음과 같이 표기하자.  
  
  $$ x_{i}^{(d)} $$  
  
  이 때의 Single Perceptron은 다음과 같다.  
  <img src="http://artrointel.github.io/assets/projects/neural-network/perceptron.JPG" />  
  
  위의 그림에서 다음 Layer의 $ x_{i}^{(1)} $를 일반화하면 다음과 같다.  
    
  $$ f(\sum_{j=1}^N w_{i,j}^{(1)} x_{j}^{(0)} + b_{i}^{(1)}) = x_{i}^{(1)} \text{, f() is an activation function. } $$
    
  이 때, 편의 상 $ z_{i}^{(1)} = \sum_{j=1}^N w_{i,j}^{(1)} x_{j}^{(0)} + b_{i}^{(1)} $ 라 정의하자.  
  
  그러면 $ f(z_{i}^{(1)}) = x_{i}^{(1)} $. 이며, 간단히 $ f(\vec z) = \vec x$ 로 표기할 수 있다.  
  
  이 처럼 이전 layer로부터 다음 layer로의 계산을 통해 $x_{i}^{d}$로 값을 전달하는 과정을 **forward propagation**이라 한다.
    
  단, $ z_{i}^{(1)} $ 의 계산에 $ x_{j}^{(0)}, 즉 이전 Layer로부터의 입력된 값을 사용한다는 데 주의한다.  
  
----------------------------------------------------------------------------------------------------------------------------------
  
### 3. Loss function (J)  
  
  위 그림의 $ d = 1 $이 output layer일 때 $ x_{i}^{(1)} $ 은 예측 값이라 할 수 있다.  
  따라서 정답 값 $ t_{i} $와 예측 값$ x_{i}^{(1)} $의 오차(이하 손실률 J)를 계산하고, 정답으로 근사시킬 $w(weight) $ 및 $ b(bias)$의 값을 찾아가야 한다.  
  이를 위해서는 $ x_{i}^{(1)} $의 계산에 참여한 $w_{i}^{(1)}$ 및 $b_{i}^{(1)}$ 값이 손실률에 얼마나 기여했는지에 대해 찾아서 적절히 update해야 할 것이다.  
<br>
$$ w \leftarrow w - \mu \frac{\delta J}{\delta w} , \mu \text{ is learning rate} $$  
<br>
  예를 들어 손실함수 J를 아래의 mean-square인 경우를 생각해보자.  
<br>
$$ J = \frac{1}{2} \sum_{i=1}^N (t_{i} - x_{i})^2 $$  
<br>
  이 경우 그래프는 대략 아래와 같다.  
  
  (ppt작업 첨부)  
  
  위의 mean-square function에서 역시 $ t_{i}=x_{i} $로 수렴하는 순간 손실률 $J = 0$이 된다.  
  매 계산마다 $J$ 값의 변화율에 기여한 $\frac{\delta J}{\delta w}$ 및 $\frac{\delta J}{\delta b}$를 계산하여 각 $w$와 $b$값들을 update할 것이다.  

----------------------------------------------------------------------------------------------------------------------------------
  
### 4. Activation functions  
  앞서 Universal Approximation Theorem에서 $\Phi$가 연속한 단조증가함수로 정의하였으니,  
  activation function으로 아래와 같은 적절한 함수를 지정할 수 있다.  
  
  * sigmoid  
  $ f(x) = \frac{1}{1+e^{-x}} $
  (그래프)  
  
  * ReLU  
  $ f(x) = $
  (그래프 첨부)
  
----------------------------------------------------------------------------------------------------------------------------------

## Source code Implementation for the theories

