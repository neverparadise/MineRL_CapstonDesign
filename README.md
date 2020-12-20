# MineRL_CapstonDesign
강화학습을 통해 마인크래프트 환경에서 문제를 해결하는 에이전트를 개발한다

과제 목표 : 랜덤하게 생성되는 세계에서 다이아몬드의 위치까지 도달하는 것

# 1. 마인크래프트 강화학습 환경
![image](https://user-images.githubusercontent.com/51039267/102716799-79b7c300-4321-11eb-8b79-d4a64210fe85.png)

마인크래프트 강화학습 환경은 마인크래프트 내의 여러가지 task를 보상을 통해서 학습시킬 수 있게 해준다.
그림의 경우 플레이어가 울타리 내의 랜덤한 위치에 생성되어 돼지를 찾아가도록 하는 예시이다.

![image](https://user-images.githubusercontent.com/51039267/102716874-e59a2b80-4321-11eb-8e51-bb46f8dd5b92.png)

위의 그림은 환경이 에이전트에게 제공하는 데이터로 64x64x3의 이미지 배열 값과, 나침반 각도 값이다. 이 정보만을 
이용해서 에이전트는 목표지점까지 도달해야 한다.

![image](https://user-images.githubusercontent.com/51039267/102716910-24c87c80-4322-11eb-815d-7438a72d568a.png)

![image](https://user-images.githubusercontent.com/51039267/102716916-30b43e80-4322-11eb-8775-a65014d9f3b1.png)

현재 학습하고자 하는 환경은 다이아몬드의 위치까지 도달하는 것이다. 목표지점까지 가까워질수록 보상을 많이 받고 멀어질수록 적게 받는다.

# 2. 데이터 전처리

![image](https://user-images.githubusercontent.com/51039267/102716927-56d9de80-4322-11eb-85d7-45f67380f44a.png)

![image](https://user-images.githubusercontent.com/51039267/102716982-b59f5800-4322-11eb-9dc8-a157f961f474.png)

데이터 전처리는 나침반 각도로 값을 가득 채운 64x64x1 텐서배열을 기존의 RGB 배열과 합치고 정규화를 진행한다.
그 결과, 64x64x4 텐서가 생성이 되며, 텐서의 차원 순서를 변경해주어야 한다. 그 이유는 pytorch의 CNN은 채널을 가장 앞의
입력으로 받기 때문이다. torch.permute()를 활용 (64x64x4) -> (4x64x64)

# 3. DQN의 구현

![image](https://user-images.githubusercontent.com/51039267/102717021-f8f9c680-4322-11eb-8dbb-8eba41548bc2.png)

![image](https://user-images.githubusercontent.com/51039267/102717036-0dd65a00-4323-11eb-8f4e-f6a908de3d26.png)

DQN의 구조는 위와 같다. 이전에 converter를 통해 변환한 텐서를 DQN 모델에 입력으로 넣으면 DQN은 Convolution Layer를 통해
각 행동에 대한 행동가치값을 반환하게 된다. 이 중에서 가장 값이 높은 것을 torch.argmax를 통해 인덱스로 삼고, 에이전트에게 전달해서
그 행동을 하게 만든다.




# Before executing the code, we need to install Pytorch, MineRL pakages

pip install minerl

To install pytorch, visit at https://pytorch.org/
