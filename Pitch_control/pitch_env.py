import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PitchControlEnv(gym.Env):
    def __init__(self):
        self.deltaE_mag = 0.01 * math.pi / 180
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'rk4'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 5 * math.pi / 180
        self.deltaEle_threshold_radians = 25 * math.pi / 180

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.theta_threshold_radians * 2, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.deltaEle = 0

    def step(self, action, theta_limit_max, theta_limit_min, theta_cmd):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        theta, Q = self.state

        # Action에 따른 Elevator 동작
        deltaE = self.deltaE_mag if action == 0 else -self.deltaE_mag
        self.deltaEle += deltaE
        if self.deltaEle >= self.deltaEle_threshold_radians:
            self.deltaEle = self.deltaEle_threshold_radians
        elif self.deltaEle <= -self.deltaEle_threshold_radians:
            self.deltaEle = self.deltaEle_threshold_radians
        else:
            self.deltaEle =self.deltaEle

        # Equation
        thetadot = Q
        Qdot = -theta + Q - self.deltaEle

        if self.kinematics_integrator == 'rk4':
            theta_k1 = thetadot
            theta_k2 = thetadot + self.tau * theta_k1 /2
            theta_k3 = thetadot + self.tau * theta_k2 /2
            theta_k4 = thetadot + self.tau * theta_k3
            theta_next = theta + self.tau * (theta_k1 + 2 * theta_k2 + 2 * theta_k3 + theta_k4) / 6

            Q_k1 = Qdot
            Q_k2 = Qdot + self.tau * Q_k1 /2
            Q_k3 = Qdot + self.tau * Q_k2 /2
            Q_k4 = Qdot + self.tau * Q_k3
            Q_next = Q + self.tau * (Q_k1 + 2 * Q_k2 + 2 * Q_k3 + Q_k4) / 6

        self.state = (theta_next, Q_next)

        # 제한 조건
        done = bool(theta < theta_limit_min or theta > theta_limit_max)

        if not done:
            if abs(theta_cmd - theta) > abs(theta_cmd - theta_next):
                reward = 1
            else:
                reward = -1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            if abs(theta_cmd - theta) > abs(theta_cmd - theta_next):
                reward = 1
            else:
                reward = -1
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done

    def reset(self):
        self.state = (np.random.uniform(low=-self.theta_threshold_radians, high=self.theta_threshold_radians), 0)
        self.steps_beyond_done = None
        return np.array(self.state)