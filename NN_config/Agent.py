import random
import numpy as np
from collections import deque
from keras.optimizers import Adam
from keras import models
from keras import layers

# 카트폴 예제에서의 DQN 에이전트
class DQNController:
    def __init__(self, state_size, action_size, load_flag):
        self.render = False
        self.load_model = load_flag
        if self.load_model:
            self.epsilon = 1.0
        else:
            self.epsilon = 1.0

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 256
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=4000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            # self.model.load_weights("./save_model/DQNPsi30v3.h5")
            self.model.load_weights("./save_model/DQNPsi10v8.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        input_tensor = layers.Input(shape=(self.state_size, ))
        x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(input_tensor)
        x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
        x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
        x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
        output_tensor = layers.Dense(self.action_size, activation='relu', kernel_initializer='he_uniform')(x)

        model = models.Model(inputs=input_tensor, outputs=output_tensor)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), random.randrange(self.action_size)
        else:
            tempshape = np.reshape(state, [1, self.state_size])
            q_value = self.model.predict(tempshape)
        return np.argmax(q_value[0, 0:2]), np.argmax(q_value[0, 3:5])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            tempshape = mini_batch[i][0]
            tempshape = np.reshape(tempshape, [1, self.state_size])
            states[i] = tempshape #mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            tempshapeN = mini_batch[i][3]
            tempshapeN = np.reshape(tempshapeN, [1, self.state_size])
            next_states[i] = tempshapeN #mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
