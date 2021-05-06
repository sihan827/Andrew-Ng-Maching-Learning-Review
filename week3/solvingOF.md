<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# Regularization - Solving the Problem of Overfitting

## 과적합(Overfitting)의 문제
경사하강법으로 회귀를 진행하면서 발생할 수 있는 문제 중 가장 큰 문제가 과적합이다. 과적합이란 쉽게 말하면 회귀 그래프가 훈련 데이터 세트에 너무 심하게 적합하도록 그려지는 것을 말한다. 반대로 과소적합은 회귀 그래프가 충분한 경사하강법 반복을 진행하지 않아서 데이터 세트에 맞지 않는 것을 말한다.

![선형 회귀 과적합](/week3/image/linearROF.png)

위의 그림은 강의자료에서 선형회귀의 과적합과 과소적합 문제의 예시를 보여준다. 맨 오른쪽 그래프처럼 회귀 그래프가 모든 데이터 점을 지나지만 새로운 입력 데이터에 대해서는 현실과 다른 부적절한 값을 예측해 내는 경우를 과적합이라고 한다.

![로지스틱 회귀 과적합](/week3/image/logisticROF.png)

위의 그림은 강의자료에서 로지스틱 회귀의 과적합과 과소적합 문제의 예시를 보여준다. 결정 경계가 중간 그래프처럼 적당히 그어져야 하는데 맨 오른쪽 그림처럼 그어질 경우 새로운 입력 데이터를 엉뚱한 클래스로 예측할 수 있다.

과적합을 해결할 수 있는 해결책은 다음과 같다.

1. 특성의 수를 줄인다. 남길 특성은 직접 고르거나 선택 알고리즘을 이용할 수 있다.
2. Regularization을 이용한다. 이 방법은 모든 특성을 다 이용하지만 $\theta$의 영향력을 어느 정도 줄여서 과적합의 발생 속도를 확연히 줄인다. 이는 엄청 많은 특성의 회귀에서도 잘 작동한다.

이번 강의에서는 Regularization 기법에 대하여 설명하고 있다.

## Regularization이 적용된 비용 함수
선형 회귀 과적합에 대한 다음 예시를 다시 보자.

![접근](/week3/image/intuition.png)

현재 $x^3$, $x^4$ 항에 의해 그래프가 구불구불해져서 과적합을 유발하고 있다. 따라서 $\theta_3$, $\theta_4$의 영향력을 줄이고 싶다. 이 경우 어떻게 해야 할까?

강의에서는 비용함수에 간단히 다음과 같이 더함으로써 해결할 수 있다고 소개한다.

$J(\theta)={1\over2m}\sum_{i=1}^m (h_{(\theta)}(x^{(i)})-y^{(i)})^2+1000\theta_3{^2}+1000\theta_4{^2}$

추가 항을 더함으로써 $\theta_3$, $\theta_4$에 페널티를 줘서 경사하강법 시 $\theta_3$, $\theta_4$의 값이 매우 작아지도록 한다. 

위의 아이디어를 $\theta$ 전체에 적용한다. 페널티는 $\lambda$로 표기한다. 식은 다음과 같다. 

$J(\theta)={1\over2m}[\sum_{i=1}^m (h_{(\theta)}(x^{(i)})-y^{(i)})^2+\sum_{j=1}^n\theta_j^2]$

중요한 것은 뒤의 regularization 항에서 $\theta_0$이 제외된 것이다. 이는 $\theta_0$이 있을 때와 없을 때 학습 결과를 비교해 볼 때 없을 때 결과가 더 좋기 때문이라고 한다. 

이렇게 완성한 regularization 항을 통해 $\theta$의 영향력을 줄임으로써 과적합을 줄이고 천천히 학습하도록 한다.

이제 남은 문제는 $\lambda$ 값을 선택하는 것이다. $\lambda$ 값이 너무 크면 $\theta$의 영향력을 너무 축소하게 되고 그러면 학습속도가 너무 느려져서 과적합 대신 과소적합이 발생할 것이다. 반대로 $\lambda$ 값이 너무 작으면 $\theta$의 영향력을 억제하지 못해서 과적합을 전혀 줄이지 못할 것이다.

## Regularization을 적용한 선형 회귀
Regularization 기법을 적용한 선형회귀의 비용함수는 위에서 소개한 것과 같다. 

$J(\theta)={1\over2m}[\sum_{i=1}^m (h_{(\theta)}(x^{(i)})-y^{1(i)})^2+\sum_{j=1}^n\theta_j^2]$

이 식에 대하여 경사하강법을 적용하면 다음과 같이 $\theta_0$만 제외하고 regularization 항도 각 $\theta_j$에 대하여 편미분이 될 것이다.

$J$가 수렴할 때까지 계속 반복 :

$\theta_0 := \theta_0-\alpha{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$

$\theta_j := \theta_j-\alpha[{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+{\lambda\over m}\theta_j]$\
($j=1,2,\cdots , n$)

선형회귀를 역행렬로 구하는 방식인 Normal Equation도 regularization을 추가할 수 있다. 기존의 Normal Equation은 다음과 같다.

$\theta=(X^TX)^{-1}X^Ty$

regularization 항을 추가하면 다음과 같다.

$\theta=(X^TX+\lambda\begin{bmatrix}0&0&\cdots&0\\0&1&\cdots&0\\\vdots&0&\ddots&0\\0&0&\cdots&1\end{bmatrix})^{-1}X^Ty$

## Regularization을 추가한 로지스틱 회귀
로지스틱 회귀에 regularization을 적용하면 비용함수만 다르고 regularization 항 자체는 같다. 따라서 비용함수식은 다음과 같다.

$J(\theta)={1\over m}\sum_{i=1}^m [-y^{(i)}log(h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+{1\over2m}\sum_{j=1}^n\theta_j^2$

경사하강법 알고리즘의 경우 비용함수 편미분시 선형회귀 비용함수 편미분과 유사하므로 regularization 항을 넣은 경사하강법 알고리즘도 선형회귀와 유사한 형태이다. 즉 다음과 같다.

$J$가 수렴할 때까지 계속 반복 :

$\theta_0 := \theta_0-\alpha{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$

$\theta_j := \theta_j-\alpha[{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+{\lambda\over m}\theta_j]$\
($j=1,2,\cdots , n$)