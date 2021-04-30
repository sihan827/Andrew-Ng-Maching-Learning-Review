<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# 변수 1개에 대한 선형 회귀 - Parameter Learning

## 경사하강법
$h_{(\theta)}(x)=\theta_0+\theta_1x$에서 
$\theta_0$, $\theta_1$ 값을 다양하게 바꿔가면서 비용 함수 $J$의 최솟값을 찾을 수도 있겠지만 너무 비효율적이다. 더 직관적으로 최솟값을 찾기 위해 나온 방법이 바로 경사하강법이다.\
![경사하강법](/week1/image/graddesinfo.png)
위 그림처럼 $\theta_0$, $\theta_1$을 해당 위치에서의 $J$에 대한 기울기 벡터를 이용해 해당 벡터 반대 방향으로 조금 이동하는 단계를 여러 번 반복하면 오목한 부분으로 $J$가 수렴할 것이다. 이를 식으로 표현하면 다음과 같다.

$\theta_j:=\theta_j-\alpha{\partial\over\partial\theta_j}J(\theta_0, \theta_1)$, $(j=0, 1)$

위 식을 $J$가 수렴할 때까지 반복한다. $J$에 대하여 각 $\theta$에 대한 편미분으로 해당 위치에서의 기울기 벡터를 구한 후 그 반대방향으로 조금씩 이동하면 최솟값을 향해 위치가 이동할 것이다. 위치가 최솟값에 가까워질수록 기울기는 0에 수렴하므로 $\theta$ 또한 특정 값에 수렴한다.

식에서 $\alpha$는 학습율(Learning Rate)이라고 하며 다음 점으로 이동할 때 얼마나 이동할 것인지에 대한 상수이다. $\alpha$의 크기 또한 주의해서 정할 필요가 있다.\
![알파 값 설정](/week1/image/alphachoose.png)
- $\alpha$가 너무 작을 시 강의자료의 위의 그림처럼 최솟값을 향해 너무 조금씩 이동하여 시간이 오래 걸린다.
- $\alpha$가 너무 클 시 강의자료의 아래 그림처럼 최솟값을 계속 뛰어넘어서 수렴하지 못하고 발산할 수도 있다.

따라서 적절한 $\alpha$값을 정하는 것이 중요하다. 

## 선형 회귀에서의 경사하강법
경사하강법을 선형회귀에 적용시키기 위해 경사하강법의 편미분 항에 선형회귀의 비용함수 $J$를 대입하여 정리하면 다음과 같다.

${\partial\over\partial\theta_j}J(\theta_0, \theta_1)={\partial\over\partial\theta_j}{1\over2m}\sum_{i=1}^m (\theta_0+\theta_1 x^{(i)}-y^{(i)})^2$ 

위 식을 $\theta_0$, $\theta_1$ 각각에 대하여 편미분하면 다음과 같다.

${\partial\over\partial\theta_0}J(\theta_0, \theta_1)={1\over m}\sum_{i=1}^m (\theta_0+\theta_1 x^{(i)}-y^{(i)})$

${\partial\over\partial\theta_1}J(\theta_0, \theta_1)={1\over m}\sum_{i=1}^m (\theta_0+\theta_1 x^{(i)}-y^{(i)})x^{(i)}$

이 항을 기존의 경사하강법 공식에 대입하면 선형회귀에서의 경사하강법 공식이 완성된다.

$J$가 수렴할 때까지 다음을 반복:

$\theta_0:=\theta_0-\alpha{1\over m}\sum_{i=1}^m (\theta_0+\theta_1 x^{(i)}-y^{(i)})$

$\theta_1:=\theta_1-\alpha{1\over m}\sum_{i=1}^m (\theta_0+\theta_1 x^{(i)}-y^{(i)})x^{(i)}$

이를 지난 강의에서 사용했던 집 가격 문제에 대입할 수 있다.\
$\theta_0=900$, $\theta_1=-0.1$부터 시작하여 어떤 $\alpha$값에 대하여 경사하강법을 진행하면 강의자료의 아래 그림처럼 진행된다.\
![실제 경사하강법 적용](/week1/image/realgraddesc.png)
그림에서 등고선 위의 빨간 x처럼 진행되다가 종국에는 $J$가 최소가 되는 $\theta$값들에 도달한다.

## 배치 경사하강법(Batch Gradient Descent)
경사하강법을 진행할 때 비용함수의 평균을 낼 때 모든 훈련 데이터들을 다 사용하여 구할 시 이를 배치 경사하강법이라고 정의한다.
