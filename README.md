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
그 행동을 하게 만든다. 따라서 DQN의 실질적인 구현은 세 개의 Convolution Layer와 한 개의 Linear Layer로 간단하게 구성할 수 있다. 


# 4. Experience Replay 

![image](https://user-images.githubusercontent.com/51039267/102717218-3f9bf080-4324-11eb-8de1-ed2e26da48a1.png)

![image](https://user-images.githubusercontent.com/51039267/102717213-33b02e80-4324-11eb-916b-9270cac4549e.png)

딥마인드에서는 DQN 모델의 성능을 향상시키기 위해 Experience Replay라는 것을 사용한다. 상태전이(transition) 들은 (상태 s, 행동 a, 보상 r, 다음 상태 s`)의 튜플로 구성할 수 있다. 상태 s에서 행동 a를 해서 보상 r을 받고 다음 상태 s`로 바뀌었다는 말이다. 
이 상태전이들은 Replay Buffer 라는 기억보관소에 저장하여 모델을 학습시킬 때 랜덤하게 기억들을 sampling하게 된다. 가장 최근의 데이터 n개를 저장하고 새로운 데이터가 들어오면 제일 오래된 데이터를 삭제한다. collections의 deque로 구현한다. 

# 5. 하이퍼파라미터 설정

![image](https://user-images.githubusercontent.com/51039267/102717239-5cd0bf00-4324-11eb-98af-1655351d0076.png)

모델을 학습시키기 위한 하이퍼 파라미터는 위와 같이 설정했다. 

# 6. 학습 결과 시각화

![image](https://user-images.githubusercontent.com/51039267/102717261-85f14f80-4324-11eb-9475-f11c020f30f1.png)

학습결과를 보면 Loss와 Reward가 모두 진동하는 양상을 보인다. 리워드가 50에 가까워지면 목적지 근처에 도달한 것이다. 몇몇 에피소드에서는 목적지에 도달하지만 대부분의 경우 도달하지 못했다. 모델이 학습을 잘 했다면 에피소드가 진행될수록 음수인 보상을 받는 경우는 줄어들 것이다. 

# 7. 원인 분석

![image](https://user-images.githubusercontent.com/51039267/102717299-c6e96400-4324-11eb-8716-96f6c8ef415e.png)

위의 그림을 보면 수풀 속에 다이아몬드가 보인다. 이렇게 목적지에 잘 도착하는 경우는 드물다. 그 원인은 분석해보면 아래의 그림과 같다. 

![image](https://user-images.githubusercontent.com/51039267/102717322-ea141380-4324-11eb-9b6d-6ee4f9aa8169.png)

대부분의 학습 과정에서 에이전트는 나무나 벽에 막혀 앞으로 나아가지 못하는 상황을 겪었다. 또는 구덩이에 빠져 허우적 대는 모습을 볼 수 있었다. 
이렇게 에이전트가 문제를 해결하지 못하는 이유를 분석해보면

1) 코드에서 정의한 에이전트의 Action Space가 문제를 풀기에 적합하지 않을 수 있다. 현재 에이전트의 행동은 앞으로 가기 (0), 왼쪽 보기 (1), 오른쪽 보기 (2)로 구성되어 있다. 모델의 학습속도를 단축시키기 위해 항상 점프와 공격을 하도록 설정했다. 하지만 이렇게 행동을 설정하면 특정 블록을 부술 수 없다. 계속 점프를 하면서 다른 블록가 번갈아 공격을 하기 때문이다. 

2) 모델 구조에 근본적으로 한계가 있다. DQN의 경우 FOMDP(Fully Observable Markov Decision Process)와 결정적인 환경에서 잘 동작한다고 알려져 있다. 마인크래프트 환경의 경우 매 에피소드마다 다른 정보를 에이전트에게 제공하기 때문에, 확률적 환경이다. 

3) 트레이닝 시간이 충분하지 않았을 수도 있다. 1000 번의 에피소드를 학습하는데 현재 Razen 3600, 32GB RAM, Nividia RTX 2070 Super 사양으로 약 5일의 시간이 걸렸다. 더 많은 학습을 진행해야했을 수도 있다. 

# 8. 개선책
1) 모델에 Recurrent Unit이 있는 ConvLSTM, 또는 ConvGRU를 사용해본다. 이를 통해 과거의 정보를 활용할 수 있는 모델을 만들어본다.

2) Action Space를 적합하게 다시 설계 한다. 하지만 이렇게 만들면 파라미터 수가 증가하기 때문에 모델 학습에 더 오랜 시간이 걸릴 것이다. 

3) Value-based Model이 아닌 Policy gradient 기반의 모델을 사용해본다. policy란 강화학습에서 정책으로 에이전트가 특정 상태에서 수행할 수 있는 행동에 대한 확률분포를 의미한다. 때문에 확률적 환경에서는 policy gradient 기반의 알고리즘이 더 잘 작동할 수 있다. 

# 9. 향후 계획

1) GAN 및 Imitation Learning, Inverse Reinforcement Learning 스터디 진행

2) Policy gradient 기반 알고리즘 스터디 진행

3) Distributional RL 논문 스터디 진행

4) 논문의 구현 및 마인크래프트 에이전트에 적용

5) 클라우드 사용법 숙지 및 서버를 통한 머신러닝 모델 훈련 방법 익히기


# Before executing the code, we need to install Pytorch, MineRL packages, open-jdk-8, malmo

pip install minerl

pip install malmo

To install pytorch, visit at https://pytorch.org/

To install pytorch, visit at https://www.oracle.com/kr/java/technologies/javase/javase-jdk8-downloads.html


참고 : https://github.com/kimbring2/DQFD_Minecraft

