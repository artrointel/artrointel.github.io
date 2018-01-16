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

먼저 Universal Approximation Theorem을 이해할 필요가 있다.

----------------------------------------------------------------------------------------------------------------------------------

### 1. Universal Approximation Theorem [[WIKI](https://en.wikipedia.org/wiki/Universal_approximation_theorem), [Reference](http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html)]

임의의 함수 $ F(x) $가 연속한 실수공간에 존재하는 함수라 할 때, 이에 아래와 같이 근사하는 연속함수 $ f(x) $가 반드시 존재한다.  
  
$$ F(x) ~= \sum_{i=1}^N v_i \Phi(w_i^Tx + b_i) = f(x) ~~~ where ~~~ i \in 1, ..., N $$
  
이 때 $ \Phi() $는 비상수인 단조증가연속함수이며, 충분히 작은 $ \epsilon > 0 $에 대하여 $ |F(x) -  f(x)| > \epsilon $을 만족시키는  
자연수 $ N $ 과 실수 $ v_i, b_i $ 및 벡터 $ w_i $가 존재한다.  


(특히, $ \Phi() $가 sigmoid 함수인 경우 [시벤코 정리](https://ko.wikipedia.org/wiki/%EC%8B%9C%EB%B2%A4%EC%BD%94_%EC%A0%95%EB%A6%AC)라 한다.)

정리에 의해 임의의 함수 $ F(x) $는 Single Perceptron만으로 $ F(x) $에 근사하는 임의의 함수 $ f(x) $를 만들어낼 수 있다는 것을 의미한다.  

----------------------------------------------------------------------------------------------------------------------------------

### 2. Single Perceptron  
  * Notation  
  아래와 같이 d번째 Layer에 존재하는 임의의 index i를 가지는 $ x $를 다음과 같이 표기한다.  
  $$ x_{i}^{(d)} $$  
  이 때의 Single Perceptron은 다음과 같다.  
  <img src="http://artrointel.github.io/assets/projects/neural-network/perceptron.JPG" />  
  
  위의 그림에서 다음 Layer의 $x_{i}^{(1)}$를 일반화하면 다음과 같다.  
    
  $$ f(\sum_{j=1}^N w_{i,j}^{(1)} x_{j}^{(0)} + b_{i}^{(1)}) = x_{i}^{(1)}, ~~~ f() is ~ an ~ activation ~ function. $$
    
  이 때, 편의 상 $ z_{i}^{(1)} = \sum_{j=1}^N w_{i,j}^{(1)} x_{j}^{(0)} + b_{i}^{(1)} $ 라 정의하자.  
  
  그러면 $ f(z_{i}^{(1)}) = x_{i}^{(1)} $. 이며, 
  간단히 $ f(\vec z) = \vec x$ 로 표기한다.  
    
  단, $ z_{i}^{(1)} $ 의 계산에 $ x_{j}^({0})$, 즉 이전 Layer로부터의 입력된 값을 사용한다는 데 주의한다.  
  
### 3. Loss function (J)  
  
  $ d = 1 $가 output layer이라면, $ x_{i}^{(1)} $ 은 예측 값이라 할 수 있다.  
  따라서 정답 $ t_{i} $에 대하여 예측 값에 대한 오차를 계산하고, 실제 값에 근사하도록 하는 $w(weight) $ 및 $ b(bias)$의 값을 찾아가야 한다.
  이를 위해서는 perceptron에 참여한 $w$ 및$b$값이, 발생한 오차에 얼마나 기여했는지에 대해 찾아서 해당 값 만큼 적절히 업데이트해야 한다.
  
$$ w \leftarrow w - \mu \frac{\delta J}{\delta w} $$
  
  예를 들어 손실함수 J()를 mean-square인 경우를 생각해보자.  
  
$$ J = \frac{1}{2} (\sum_{i=1}^N t_{i} - x_{i})^2 $$
  이 경우 그래프는 대략 아래와 같다.  
  
  <ppt작업>  
  
  위의 2차 곡선으로 그래프가 그려질 때, $t=x$로 정답값에 수렴하는 순간 손실률 $J() = 0$이 된다.  
  수렴을 위해 $J$ 값의 변화율 만큼의 변화를 $w$ 및 $b$ update에 활용한다.  
  정답 $t$에 근사할수록 변화율이 줄어드므로, 근사가 반복될 수록 t에 수렴하게 될 것임을 유추할 수 있다.  

### 4. Foward propagation & Activation functions
  
  
------------------------------------------------------------

## Source code Implementation for the theories

