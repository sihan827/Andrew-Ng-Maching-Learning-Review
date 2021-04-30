<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# 다변수 선형 회귀 - Normal Equation

## Normal Equation이란?
경사하강법은 정확한 $\theta$를 바로 구하는 것이 아니라 미리 정한 여러 번의 단계를 거쳐서 $J(\theta)$의 값이 최소에 수렴할 즈음의 $\theta$를 계산한다. Normal Equation은 경사하강법과 다르게 $J(\theta)$가 최소가 되는 정확한 $\theta$를 바로 계산한다. 

Normal Equation은 데이터를 벡터화하여 행렬로 표현한 뒤 행렬의 연산으로 바로 $\theta$를 구해 낸다. 

먼저 $n$개의 특성을 가진 $m$쌍의 데이터 쌍 $(x^{(1)}, y^{(1)})$, $(x^{(2)}, y^{(2)})$, $\cdots$ ,$(x^{(m)}, y^{(m)})$를 벡터화하여 행렬로 표현한다. $x^{(i)}$는 $x_0^{(i)}(=1)$, $x_1^{(i)}$, $x_2^{(i)}$, $\cdots$ , $x_n^{(i)}$으로 이루어진 벡터이다. ($\theta_0$ 항 계산을 위해 $x_0^{(i)}(=1)$을 모든 데이터 벡터의 1행에 추가한다.)이를 묶으면 전체 데이터를 다음과 같이 $m\times (n+1)$ 행렬 $X$로 표현할 수 있다. 

$X=\begin{bmatrix}(x^{(1)})^T\\(x^{(2)})^T\\\vdots\\(x^{(m)})^T\end{bmatrix}$

결과값 $y$는 $m$차원 벡터로 표현할 수 있다.

$y=\begin{bmatrix}y^{(1)}\\y^{(2)}\\\vdots\\y^{(m)}\end{bmatrix}$

그러면 위 데이터 행렬들에 대하여 계수에 대한 $(n+1)$차원 벡터 $\theta$를 다음과 같이 표현할 수 있다.

$\theta=\begin{bmatrix}\theta_0\\\theta_1\\\theta_2\\\vdots\\\theta_n\end{bmatrix}$

이제 이 행렬들의 곱으로 다변수 선형회귀를 표현할 수 있다. 기존의 가설처럼 행렬을 배치하면 $X\theta$=y로 표현된다. 즉 해당 행렬식을 잘 분석해서 $\theta$를 구할 수 있다.

일단 역행렬을 이용하기 위해 $m\times (n+1)$ 행렬 $X$를 정사각행렬로 만들 필요가 있다. 간단하게 $X^T$를 양변에 곱하면 $X^TX$는 $(n+1)\times (n+1)$ 행렬이 되므로 역행렬이 존재할 수도 있게 된다. 그러면 다음과 같이 양변에 $X^TX$의 역행렬을 곱해서 다음과 같이 $\theta$를 바로 구할 수 있다.

$\theta=(X^TX)^{-1}X^Ty$

그렇다면 경사하강법과 Normal Equation 중 어떤 것이 좋은 것인지 의문이 생긴다.
- 경사하강법은 $\alpha$를 선택해야 하고 많은 반복을 거쳐야 $\theta$의 근사치를 구할 수 있지만 $n$이 크더라도 잘 동작한다.
- Normal Equation은 $\alpha$를 선택할 필요가 없고 반복을 거치지 않아도 되지만 $(X^TX)^{-1}$가 $(n+1)\times (n+1)$이므로 $n$이 크면 역행렬 계산에 시간이 오래 걸린다.

일반적으로 $n$이 10000보다 크면 경사하강법을 사용하고 10000보다 작으면 Normal Equation을 사용할 만하다고 한다.

## $(X^TX)^{-1}$가 존재하지 않는다면?
행렬의 역행렬이 존재하지 않을 수 있다. 강의에서는 이를 Pseudo-inverse Matrix라는 유사 역행렬로 계산하는 방법도 있지만 보통 $X^TX$의 역행렬이 존재하지 않을 시 다음과 같은 방법을 사용해보기를 권고한다.
- 불필요한 특성을 줄인다. 모든 특성들은 선형 독립이어야 한다. 예를 들어 $x_1$은 제곱피트 단위이고 $x_2$는 제곱미터 단위라면 $1m=3.28feet$이므로 $x_1=3.28^2x_2$로 표현되므로 서로 선형 독립이 아니다. 따라서 둘 중 하나를 제거해야 한다.
- 특성 개수가 너무 많으면 (ex) $m<n$) 특성을 좀 지우거나 뒤의 강의에서 나오는 Regularization 기법을 사용해야 한다.