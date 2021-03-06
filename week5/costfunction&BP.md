<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# 인공신경망 - Cost Function and Backpropagation

## 신경망의 비용함수
신경망의 비용함수는 기본적으로 로지스틱 회귀의 비용함수와 유사하다.

이진 분류의 경우 클래스가 2개뿐이므로 0, 1 사이의 값으로 분류를 진행한다. 따라서 출력 레이어의 유닛이 하나만 있어도 된다. 이 경우 기존의 로지스틱 회귀와 비용함수가 유사할 것이다. 

다중 클래스 분류의 경우 클래스가 여러 개이므로 기존에 로지스틱 회귀로 다중 클래스 문제를 해결하던 것처럼 one-vs-all을 기반으로 분류를 진행한다. 따라서 출력 레이어의 유닛이 클래스 수와 같으므로 포워딩의 결과는 클래스 수를 K라 하면 K차원 벡터가 된다. 즉 훈련세트로 제공되는 데이터의 라벨들 또한 K차원 벡터가 되어야 한다. 출력 레이어의 각 유닛에 로지스틱 회귀의 비용함수를 적용하면 K차원 벡터의 비용함수 값이 나온다. 즉 이를 다 더하면 신경망의 비용함수가 된다. 

$L$을 신경망의 총 레이어 수, $K$를 클래스의 수, $s_l$을 레이어 l의 유닛 수, $(h_\theta(x))_p$를 출력 레이어의 p번째 유닛이라고 하면 신경망의 비용함수는 다음과 같은 식으로 나타내어진다.

$J(\theta)=-{1\over m}[\sum_{i=1}^m \sum_{k=1}^K y_k^{(i)}log((h_\theta(x^{(i)}))_k)+(1-y_k^{(i)})log(1-(h_\theta(x^{(i)}))_k)] + {\lambda\over2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}\theta_j^2$ 

뒤의 항은 로지스틱 회귀 비용함수에서도 사용하던 regularization 항으로 신경망의 모든 파라미터의 제곱을 다 더해서 평균을 내는 것이다.

## 역전파 (Backpropagation) 알고리즘
비용함수 자체는 로지스틱 회귀와 유사하다지만 이를 가지고 경사하강법으로 $\theta$를 학습하려면 모든 $\theta_{ij}^{(l)}$에 대하여 비용함수를 미분시켜야 한다. 그러나 신경망은 여러 레이어로 쌓여 있기 때문에 편미분이 어려워진다. 이 편미분을 쉽게 할 수 있도록 정리된 알고리즘이 역전파 알고리즘이다. 

먼저 순전파(Forward Propagation)은 이제까지 입력 벡터를 신경망에 넣어서 포워딩하는 것과 동일하다. 강의에서는 4개의 레이어로 구성된 신경망에 대한 순전파 알고리즘의 예시를 보여주고 있다. 다음 슬라이드는 입력 데이터 $x$를 순전파하는 식을 보여준다.

![순전파 알고리즘](/week5/image/fp.png)

출력 레이어인 레이어 4를 제외하고 bias 유닛을 추가한다는 점에 유의해야 한다.

그렇다면 역전파 알고리즘이란 무엇인가? 쉽게 보면 순전파의 반대이다. 출력 유닛부터 거꾸로 비용함수에 대하여 각 파라미터들에 대한 편미분 식을 계산해 나가는 것이다. 변수들을 순전파 알고리즘에서 사용한 것들을 사용한다.

먼저 각 출력 유닛 $a_j^{(l)}$들의 에러를 정의하는 $\delta_j^{(l)}$을 정의한다. 
- 레이어 4는 출력 레이어이므로 에러는 그냥 입력 데이터 라벨과의 차가 된다. 즉 $\delta_j^{(4)}=a_j^{(4)}-y_j$이고 벡터화시키면\
$\delta^{(4)}=a^{(4)}-y$\
이다.
- 레이어 3부터는 출력 레이어가 아니므로 다른 방식으로 구해야 한다. 일단 벡터화된 공식은\
$\delta^{(3)}=(\theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})$\
이다. 여기서 .* 연산자는 MATLAB, OCTAVE 등의 프로그래밍 언어에서 동일 차원의 행렬에서 같은 자리에 있는 원소끼리 곱하는 연산자이다. $g'(z^{(3)})$은 미분한 활성화함수에 $z^{(3)}$을 대입한 값이다. 이를 실제로 미분하면 $a^{(3)}.*(1-a^{(3)})$이 된다.
- 레이어 2 또한 레이어 3과 비슷한 형태의 공식이 나온다. 벡터화된 공식은\
$\delta^{(2)}=(\theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})$\
이다. $\delta^{(3)}$과 유사한 것을 볼 수 있다. $g'(z^{(2)})$은 미분하면 $a^{(2)}.*(1-a^{(2)})$이 된다.
- 레이어 1의 경우 입력 레이어이기 때문에 에러를 구할 필요가 없다.

이를 이용하면 훈련데이터 $(x, y)$ 하나에 대하여 비용함수 $J(\theta)$를 $\theta_{ij}^{(l)}$로 편미분한 값은 $a_j^{(l)}\delta_i^{(l+1)}$가 된다. (regularization 제외했을 때)\
즉 훈련 데이터 m개를 전부 사용한다면 m개에 대하여 $J(\theta)$를 $\theta_{ij}^{(l)}$로 편미분한 값을 모두 더해서 평균 낸 것이 실제 ${\partial\over\partial\theta_{ij}^{(l)}}J(\theta)$가 된다.  

이를 알고리즘으로 정리하면 다음과 같다. 

훈련 셋 { $(x_1, y_1), (x_2, y_2), \cdots, (x_m, y_m)$ }에 대하여
모든 $i, j, l$에 대하여 $\vartriangle_{ij}^{(l)} = 0$으로 초기화한다. 이 변수는 모든 훈련 데이터들에 대하여 편미분 값들을 축척시키려고 정의한 변수이다.\
이제 데이터 셋이 m개이므로 $p=1$부터 $m$까지 총 m번 다음을 반복한다.
- 입력 레이어 유닛 $a^{(1)} = x^{(p)}$로 초기화한다.
- 각 레이어에 대하여 $a^{(l)}$ 계산을 위해 순전파를 수행한다. (총 레이어 개수가 L개이므로 $l$은 2~L까지이다.)
- 이제 데이터의 라벨 벡터 $y^{(p)}$를 이용하여 레이어 L 유닛의 에러 $\delta^{(L)}=a^{(L)}-y^{(p)}$를 계산한다.
- $\delta^{(L)}$을 이용하여 $\delta^{(L-1)}, \delta^{(L-2)}, \cdots, \delta^{(2)}$를 계산한다. 
- 모든 $i, j, l$에 대하여 $\vartriangle_{ij}^{(l)} := \vartriangle_{ij}^{(l)} - a_j^{(l)}\delta_i^{(l+1)}$를 계산한다.

반복문을 모두 마치면 모든 $i, j, l$에 대해 $\vartriangle_{ij}^{(l)}$은 모든 훈련 셋 m개에 대하여 각 데이터를 $\theta_{ij}^{(l)}$로 편미분한 값의 총합을 갖게 된다. 이를 이용하면 비용함수를 각 파라미터로 편미분한 값을 얻을 수 있다. 아래는 미분값을 구하는 식이다. (regularization이 포함되었는데 regularization 항 자체의 편미분은 간단하므로 따로 설명이 있지는 않았다.)

$D_{ij}^{(l)} = {1\over m}\vartriangle_{ij}^{(l)} + \lambda\theta_{ij}^{(l)}$ ($j\ne0$ 일 때)

$D_{ij}^{(l)} = {1\over m}\vartriangle_{ij}^{(l)}$ ($j=0$ 일 때)

$j=0$이라는 것은 bias 항임을 의미한다. 따라서 regularization 항이 없다. (앞에서 과적합을 배울 때 배운다.)

즉 위의 $D_{ij}^{(l)}$이 바로 ${\partial\over\partial\theta_{ij}^{(l)}}J(\theta)$가 되는 것이다.

## 역전파 알고리즘 분석
강의에서는 간단한 레이어 4개짜리 신경망을 예로 역전파 알고리즘을 더 자세히 설명하고 있다.

먼저 아래 그림을 통해 순전파를 살펴보자.

![순전파](/week5/image/fpexample.png)

훈련 데이터 $(x^{(i)}, y^{(i)})$으로 순전파를 시행하면 $z_1^{(3)}$은 그림의 식처럼 계산될 것이다. 이전 레이어의 유닛들이 각각 선에 해당되는 $\theta_{ij}^{(l)}$과 곱해져서 합쳐진다. 즉 이 $\theta_{ij}^{(l)}$들이 $z_1^{(3)}$에 영향을 준다. 다른 $z$도 마찬가지로 생각할 수 있다.

그렇다면 역전파 알고리즘은 무엇을 하는가? 먼저 신경망의 비용함수 식을 가져오면 다음과 같다.

$J(\theta)=-{1\over m}[\sum_{i=1}^m \sum_{k=1}^K y_k^{(i)}log((h_\theta(x^{(i)}))_k)+(1-y_k^{(i)})log(1-(h_\theta(x^{(i)}))_k)] + {\lambda\over2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}\theta_j^2$ 

이를 분석하기 쉽게 하기 위해서 regularization 항은 없다고 하고 출력 레이어의 유닛이 하나뿐이므로 출력 유닛을 더하는 시그마는 없다. 이때 훈련 데이터 $(x^{(i)}, y^{(i)})$에 대하여 비용함수 값을 cost(i)라 하면 식은 대괄호 안의 식 그대로 쓸 수 있다. 

cost(i)$=y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))$

이제 이를 가지고 $\delta_{ij}^{(l)}$의 의미를 알아보자. 다음 슬라이드는 역전파의 과정을 보여 주고 있다.

![역전파](/week5/image/bp.png)

$\delta_{ij}^{(l)}$의 정의를 되새겨보면 $a_j^{(l)}$에 대한 비용의 에러이다. 이는 다음과 같이 표현된다고 한다. 

$\delta_{ij}^{(l)}={\partial\over\partial z_j^{(l)}}cost(i)$

즉 위의 편미분 식이 역전파 알고리즘에서 사용한 공식인 셈이다.

더 직관적으로 보면 순전파는 입력 쪽에서 출력 쪽으로 각 유닛을 계산해 갔다. 역전파는 그 반대과정이라고 생각하면 된다. 출력 쪽에서 입력 쪽으로 각 유닛의 에러인 $\delta$를 계산해 나가는 것이다. 즉 시작은 출력 레이어의 에러부터 시작한다. 예시로 든 신경망으로 $\delta_1^{(4)}, \delta_2^{(3)}, \delta_2^{(2)}$를 순차적으로 계산해 나가면 그림의 식처럼 전파될 것이다. 이를 직접 써보면 다음과 같다.

$\delta_1^{(4)} = a^{(4)}-y^{(i)}$

$\delta_2^{(3)}=\theta_{12}^{(3)}\delta_1^{(4)}$

$\delta_2^{(2)}=\theta_{12}^{(2)}\delta_1^{(3)}+\theta_{22}^{(2)}\delta_2^{(3)}$

즉 순전파로 층과 층 사이 서로 연결된 선에 해당하는 가중치를 곱하고 더해서 나아가듯이 역전파 또한 층과 층 사이 서로 연결된 선에 해당하는 가중치를 곱해서 더하는 식으로 진행된다. 주의할 점은 역전파 시에 bias 유닛의 에러 또한 역전파로 계산을 해서 미분값을 구해야 하는가에 대한 문제인데, 일단 이 강의에서는 bias 유닛에 대한 에러는 계산하지 않고 따라서 역전파 시에도 이를 이용하지 않는다. 이는 역전파 알고리즘의 구현마다 다르다고 한다.