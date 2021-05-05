<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# Vectorize

## 식의 벡터화
이제까지 강의에서 경사하강법 알고리즘은 다음과 같다.

$J(\theta)$가 수렴할 때까지 계속 반복:

$\theta_j := \theta_j-\alpha{\partial\over\partial\theta_j}J(\theta)$\
($j=0,1,2,\cdots , n$)

즉 알고리즘 상에서는 $\theta$의 원소의 수만큼 루프를 돌리는 셈이 된다. 프로그래밍 언어를 이용해 알고리즘을 진행할 때는 Octave의 행렬이나 Python의 Numpy 등 강력한 행렬 연산을 통해 저런 루프문보다 빠르게 알고리즘을 수행할 수 있다. 이를 위해서는 알고리즘을 행렬식으로 바꾸는 벡터화(Vectorization)이 필요하다. 

## 기존의 Notation
훈련 데이터 세트는 총 m개의 입력 데이터와 결과 데이터를 가지고 특성이 n개일 때 입력 데이터 벡터 $x$는 $x_0(=1), x_1, \cdots, x_n$의 특성들로 이루어진 벡터이다. 즉 i번째 입력 데이터 벡터 $x$는 다음과 같은 $(n+1)$차원 벡터이다.

$x^{(i)}=\begin{bmatrix}x_0^{(i)}\\x_1^{(i)}\\x_2^{(i)}\\\vdots\\x_n^{(i)}\end{bmatrix}$

이를 이용해 전체 훈련 데이터 세트를 행렬 $X$로 표현하면 입력 데이터 벡터마다 전치를 통해 다음과 같은 $(m\times(n+1))$차원 벡터로 표현할 수 있다.

$X=\begin{bmatrix}(x^{(1)})^T\\(x^{(2)})^T\\\vdots\\(x^{(m)})^T\end{bmatrix}$

결과 벡터 y도 다음과 같이 $(m)$차원 벡터로 표현된다.

$y=\begin{bmatrix}y^{(1)}\\y^{(2)}\\\vdots\\y^{(m)}\end{bmatrix}$

$\theta$ 벡터는 n개의 특성이 있으므로 $(n+1)$차원의 벡터이다. 다음과 같다.

$\theta=\begin{bmatrix}\theta_0\\\theta_1\\\theta_2\\\vdots\\\theta_n\end{bmatrix}$

이제 이 notation을 통해 기존의 비용함수 식은 시그마를 제외하고 행렬만으로 식을 작성할 수 있으며 경사하강법 알고리즘은 루프를 사용하지 않고 행렬식만으로 표현이 가능하다.

## 선형 회귀의 벡터화
기존의 선형 회귀 비용함수는 다음과 같다.

$J(\theta)={1\over2m}\sum_{i=1}^m (h_{(\theta)}(x^{(i)})-y^{(i)})^2$

먼저 가설 식은 $h_\theta=X\theta$로 간단하게 표현된다. $(m\times(n+1))\times((n+1)\times1)$이므로 결과는 $m$차원 벡터이다. 각 원소는 각 입력 데이터 벡터에 대한 가설 식이다. 제곱의 합을 구하는 시그마 부분은 간단하게 전치를 이용하여 행렬 식으로 표현할 수 있다. 즉 위의 비용함수는 다음과 같이 행렬식으로 간편하게 표현된다.

$J(\theta)={1\over2m} (X\theta - y)^T(X\theta - y)$

이제 경사하강법을 보면 기존의 경사하강법 식은 다음과 같다.

$J$가 수렴할 때까지 계속 반복 :

$\theta_j := \theta_j-\alpha{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$\
($j=0,1,2,\cdots , n$)

위의 식을 시그마를 풀어서 정리를 해보면 다음과 같이 간단한 행렬식으로 정리할 수 있다.

$J$가 수렴할 때까지 계속 반복 :

$\theta := \theta-\alpha{1\over m}X^T(X\theta - y)$

## 로지스틱 회귀의 벡터화
로지스틱 회귀의 가설식은 $X\theta$에 시그모이드 함수가 씌워진다. 시그모이드 함수를 $g$라고 하면 가설식 $h_\theta$는 $g(X\theta)$가 된다.

로지스틱 회귀의 비용함수는 다음과 같다.

$J(\theta)={1\over m}\sum_{i=1}^m [-y^{(i)}log(h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$ 

마찬가지로 전치를 이용해 시그마를 지운 행렬식으로 표현이 가능하다. 다음과 같다. (주의: 1은 모든 원소가 1인 $m$차원 벡터이다.)

$J(\theta)={1\over m} [-y^Tlog(g(X\theta))-(1-y)^Tlog(1-g(X\theta))]$

다음으로 경사하강법 알고리즘을 보자. 기존의 로지스틱 회귀 경사하강법 알고리즘 식은 선형회귀와 매우 유사하다.

$J$가 수렴할 때까지 계속 반복 :

$\theta_j := \theta_j-\alpha{1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$\
($j=0,1,2,\cdots , n$)

따라서 선형회귀와 유사하게 행렬식으로 나타내어진다. 다음과 같다.

$J$가 수렴할 때까지 계속 반복 :

$\theta := \theta-\alpha{1\over m}X^T(g(X\theta) - y)$