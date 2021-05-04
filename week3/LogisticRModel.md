<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# 로지스틱 회귀 - Logistic Regression Model

## 로지스틱 회귀의 비용 함수
비용 함수를 적용하기 이전에 선형 회귀에서 사용하였던 notation을 가져온다. 특성은 $n$개이고 데이터 개수가 $m$개인 훈련 데이터가 있으며 데이터 하나는 $x$ 벡터로 표기된다. 클래스(라벨) 값은 $y$로 표기된다. 정리하면 다음과 같다.

훈련 데이터 세트 : {$(x^{1}, y^{1}), (x^{2}, y^{2}), \cdots ,(x^{m}, y^{m})$}

입력 데이터 벡터 $x$ : $\begin{bmatrix}x_0\\x_1\\\vdots\\x_n\end{bmatrix}$ ($x_0=1$)

출력 라벨 y의 종류 : $y\in$ {$0, 1$}

가설 함수 : $h_\theta(x)={1\over 1+e^{-\theta^Tx}}$

이제 비용 함수를 적용해야 한다. 이전에 선형 회귀에서 사용하던 MSE를 그대로 사용해도 로지스틱 회귀에서 비용의 최솟값을 찾을 수 있는가를 생각해 볼 필요가 있다. 결론부터 말하자면 찾기 어려워진다. 그 이유는 시그모이드 함수에서 찾을 수 있다. 먼저 기존의 MSE를 적용한다고 해보자. 기존의 MSE 비용함수의 식은 다음과 같다.

비용함수 : $J(\theta)={1\over2m}\sum_{i=1}^m (h_{(\theta)}(x^{(i)})-y^{(i)})^2$ 

선형 회귀에서는 $h_{(\theta)}(x^{(i)})$가 $\theta$의 원소들을 계수로 가지는 다항식이므로 비용함수 $J(\theta)$는 $\theta$의 원소에 관한 2차식이다. 이를 그래프로 표현하면 강의자료의 다음 그림과 같이 볼록 함수가 된다.

<img src="/week3/image/convex.png" width="40%" height="30%" title="Convex"></img>

따라서 경사하강법으로 최소점을 따라서 이동하면 된다. 

반면에 로지스틱 회귀에서는 다항식에 라벨의 확률 계산을 위해 시그모이드 함수를 덧씌운다. 따라서 $\theta$의 원소에 대한 이차식이 아닌 더 복잡한 식이 된다. 이를 그래프로 표현하면 강의자료의 다음 그림과 같이 볼록 함수가 아닌 더 복잡한 함수 그래프가 그려질 것이며 곳곳에 Local Minimum이라는 함정이 존재하게 된다.

<img src="/week3/image/nonconvex.png" width="40%" height="30%" title="Non-convex"></img>

따라서 경사하강법으로 최소를 향해 이동하기에 어려움이 생긴다. 따라서 다른 비용함수가 필요하다.

먼저 대략적으로 생각하면 모든 회귀에 대해서 다음과 같은 생각이 적용된다. 입력에 대하여 출력이 정답과 가까워지면 비용 함수의 값이 감소하는 방향으로 가야 한다. 이를 로지스틱 회귀에 적용하면 라벨이 1이면 출력값이 1에 가까워질수록 비용함수의 값이 감소해야 하고 라벨이 0이면 출력값이 0에 가까워질수록 비용함수의 값이 감소해야 한다. 이를 어떻게 함수로 정의할 것인가?

강의에서는 다음과 같은 비용함수를 소개한다. $y=1$일 때와 $y=0$일 때 서로 다른 비용함수를 제시한다. 먼저 $y=1$일 시에는 다음과 같은 비용함수를 사용한다.

$Cost(h_\theta(x), y)=-log(h_\theta(x))$ ($y=1$)

이 함수를 그래프로 그리면 강의자료의 다음 그림과 같다.\
![y=1](/week3/image/y=1.png)\
그래프를 보면 알 수 있듯이 값이 1에 가까워질수록 비용이 감소하고 0에 가까워질수록 비용이 급격하게 증가한다.

다음으로 $y=0$일 때는 다음과 같은 비용함수를 사용한다.

$Cost(h_\theta(x), y)=-log(1-h_\theta(x))$ ($y=0$)

이 함수를 그래프로 그리면 강의자료의 다음 그림과 같다.\
![y=0](/week3/image/y=0.png)\
그래프를 보면 알 수 있듯이 값이 0에 가까워질수록 비용이 감소하고 1에 가까워질수록 비용이 급격하게 증가한다.

위의 두 함수를 로지스틱 회귀의 비용함수로 사용하면 원하는 대로 비용함수의 최솟값에 접근할 수 있을 것이다.

## 비용함수의 정리 및 경사하강법 적용
앞에서 정의한 비용함수를 이용하여 로지스틱 회귀의 비용함수를 정의하면 다음과 같다. 

$
Cost(h_\theta(x), y)=
\begin{cases}
-log(h_\theta(x)), (y=1) \\
-log(1-h_\theta(x)), (y=0)
\end{cases}
$일 때

$J(\theta)={1\over m}\sum_{i=1}^m Cost(h_{(\theta)}(x^{(i)}), y^{(i)})$ 

이 식을 더 간편하게 하나의 식으로 정리할 수 있다. $y$는 오직 0, 1만 가능하므로 이를 이용해 비용함수를 다음과 같이 정리할 수 있다.

$J(\theta)={1\over m}\sum_{i=1}^m [-y^{(i)}log(h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$ 

이러면 $y^{(i)}=1$이면 $-log(h_\theta(x^{(i)}))$만 남고 $y^{(i)}=0$이면 $-log(1-h_\theta(x^{(i)}))$만 남으므로 적절한 비용함수가 된다. 

이제 남은 것은 이 비용함수를 최소화하는 $\theta$를 찾아서 로지스틱 회귀를 진행하는 것이다. 기본적인 경사하강법 진행 방식은 선형 회귀 때와 똑같이 $\theta$의 각 원소에 대하여 편미분을 진행하고 각 원소에 대해 학습율만큼 곱해서 빼주는 것이다. 즉 알고리즘은 다음과 같다.

$J$가 수렴할 때까지 계속 반복 :

$\theta_j := \theta_j-\alpha{\partial\over\partial\theta_j}J(\theta)$\
($j=0,1,2,\cdots , n$)

${\partial\over\partial\theta_j}J(\theta)$는 시그모이드 함수를 미분하는 것이라 어려워 보이지만 실제로 미분을 하면 다음과 같이 부분적으로 미분을 해서 구할 수 있다.

$h_\theta(x)=sigmoid(g_\theta(x))$일 때

${\partial\over\partial\theta_j}J(\theta)={\partial J(\theta)\over\partial h_\theta(x)}\times{\partial h_\theta(x)\over\partial g_\theta(x)}\times{\partial g_\theta(x)\over\partial\theta_j}$

이를 계산하면 선형회귀와 비슷한 형태의 미분 식이 나온다. 다음과 같다.

${\partial\over\partial\theta_j}J(\theta)={1\over m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

이를 경사하강법 알고리즘에 대입해서 진행하면 $J(\theta)$의 최솟값으로 점차 이동할 것이다.

## 고급 최적화 기법
이제까지 $J(\theta)$의 최솟값을 찾으려고 사용한 최적화 알고리즘은 경사하강법이다. 이외에도 사용할 수 있는 최적화 기법은 많다. 강의에서는 Conjugate gradient, BFGS, LBFGS 등의 기법이 있다고 소개하며 경사하강법과의 차이로 학습률 $\alpha$를 고를 필요가 없고 훨씬 빨리 최솟값에 가까워질 수 있지만 경사하강법에 비해 훨씬 복잡하다고 알려주었다.