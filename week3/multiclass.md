<script type="text/javascript" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>

<span style="color:pink">*이 글은 Coursera에서 Andrew Ng 교수님의 머신러닝 강의를 듣고 읽기 자료를 읽으면서 복습차원에서 개인적으로 요약한 글입니다.*<span>

# 로지스틱 회귀 - Multiclass - classification

## 다중 클래스 분류
앞의 강의에서 배운 것은 클래스가 두 개일 때의 로지스틱 회귀였다. 그러나 실세계의 문제에는 클래스가 여러 개인 문제가 많다. 예를 들면 날씨는 Sunny, Cloudy, Rain, Snow 등 분류될 수 있는 클래스가 많다. 이 경우에는 어떻게 해야 하는가? 

답은 생각보다 간단하다. 클래스의 개수만큼 로지스틱 회귀를 진행하면 된다. 이를테면 아래의 강의자료처럼 클래스가 세 개인 분류 문제에서는 세 번의 로지스틱 회귀를 수행하면 된다.\
![다중클래스 문제](/week3/image/multiclass.png)\
그림처럼 클래스 1/클래스 2, 3 두 분류로 회귀 한번, 클래스 2/클래스 1, 3 두 분류로 회귀 한번, 클래스 3/클래스 1, 2 두 분류로 회귀 한번 하여 로지스틱 회귀를 총 3번 진행하면 $\theta$ 벡터가 총 세 개가 나오고 회귀 식도 $h_\theta^{(1)}(x)$, $h_\theta^{(2)}(x)$, $h_\theta^{(3)}(x)$ 총 세 개가 나온다. 따라서 입력 벡터 $x$를 각 식에 넣어서 가장 큰 회귀값일 때 클래스를 선택하면 된다.

위의 사례를 클래스 i에 대하여 조건확률식으로 쓰면 다음과 같다.

$h_\theta^{(i)}(x)=P(y=i|x;\theta)$

즉 다음을 만족하는 클래스 $i$를 선택한다.

$max_i h_\theta^{(i)}(x)$