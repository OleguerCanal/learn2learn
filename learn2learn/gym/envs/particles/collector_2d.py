#!/usr/bin/env python3

import cv2
import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv

class Entity2D:
    """Represents an element:
        - agent (id:-1)
        - red collectible (id:0)
        - green collectible (id:1)
        - blue collectible (id:2) 
    """

    def __init__(self, entity_type, world_size, x=None, y=None):
        self.type = entity_type
        self.world_size = world_size
        self.x = x
        self.y = y

    def set_randomly(self):
        self.x = np.random.uniform(0, self.world_size)
        self.y = np.random.uniform(0, self.world_size)

    def distance(self, other):
        return np.linalg.norm([self.x - other.x, self.y - other.y])

    def step(self, delta_x, delta_y):
        self.x = np.clip(self.x + delta_x, 0, self.world_size)
        self.y = np.clip(self.y + delta_y, 0, self.world_size)

    def get_coord(self, shape):
        x = shape[1]*self.x/float(self.world_size)
        y = shape[0]*self.y/float(self.world_size)
        return tuple((int(x), int(y)))

    def set_pos(self, coord, shape):
        """If the world was of the shape given, set the agent to its position 
        """
        self.x = self.world_size*coord[1]/float(shape[1])
        self.y = self.world_size*coord[0]/float(shape[0])

    def __str__(self):
        return "Entity2D type: " + str(self.type)


class Collector2DEnv(MetaEnv):
    """
    **Description**

    Each task is defined by the reward obtained for
    collecting each of the goals [reward_red, reward_green, reward_blue]
    """

    def __init__(self, max_steps=500, size=200, task=None):
        self.seed()
        super(Collector2DEnv, self).__init__(task)
        self.max_steps = max_steps
        self.size = size
        self.step_size = 10
        self.distance_threshold = 5
        self.movements = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # Define environment entities
        self.agent = Entity2D(world_size=size, entity_type=-1)
        self.entities = [Entity2D(world_size=size, entity_type=0),
                         Entity2D(world_size=size, entity_type=1),
                         Entity2D(world_size=size, entity_type=2)]


        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
                                       shape=(2,), dtype=np.float32)
        self.reset()

    # -------- MetaEnv Methods --------
    def sample_tasks(self, num_tasks):
        """
        Tasks correspond to a goal point chosen uniformly at random.
        """
        reward_weights = self.np_random.uniform(0, 1.0, size=(num_tasks, 3))
        tasks = [{'w': reward_weight} for reward_weight in reward_weights]
        return tasks

    def set_task(self, task):
        self._task = task
        self._w = task['w']

    # -------- Gym Methods --------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, env=True):
        """
        Randomly re-sets all entities
        """
        self.step_count = 0
        self.agent.set_randomly()
        for entity in self.entities:
            entity.set_randomly()
        return self.get_observation()

    def get_reward_features(self):
        reward_features = np.zeros(3, dtype=np.float32)
        normalization = float(self.max_steps*self.size)
        for entity in self.entities:
            reward_features[entity.type] -= self.agent.distance(entity)/normalization
            if self.agent.distance(entity) < self.distance_threshold:
                reward_features[entity.type] += 1.
                entity.set_randomly()
        return reward_features

    def step(self, action):
        # Move agent
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self.agent.step(delta_x=self.step_size*action[0], delta_y=self.step_size*action[1])
        
        # Count step
        self.step_count += 1
        done = (self.step_count == self.max_steps)

        # Convert reward features to reward
        reward_features = self.get_reward_features()
        reward = np.dot(reward_features, self._w)
        return self.get_observation(), reward, done, self._task

    def get_observation(self):
        def normalize(val):
            return 2*(val/self.size - 0.5)
        agent_x = normalize(self.agent.x)
        agent_y = normalize(self.agent.y)
        obs = []
        for entity in self.entities:
            obs.append((normalize(entity.x) - agent_x)/2.)
            obs.append((normalize(entity.y) - agent_y)/2.)
        obs.append(self._w[0])
        obs.append(self._w[1])
        obs.append(self._w[2])
        return np.array(obs, dtype=np.float32)


    def get_image(self, shape=(200, 200), plot_agent=True):
        img = np.ones((shape[0], shape[1], 3), dtype=np.uint8)*255
        if plot_agent:
            agent_color = (0, 0, 0)
            if task is not None:
                agent_color = tuple(
                    (255*self._w[0].item(), 255*self._w[1].item(), 255*self._w[2].item()))
            img = cv2.circle(img, self.agent.get_coord(shape=shape),
                             radius=4, color=agent_color, thickness=-1)
        for entity in self.entities:
            img = cv2.circle(img, entity.get_coord(shape=shape), radius=6,
                             color=self.colors[entity.type], thickness=3)
        return img

    def render(self, mode='human', close=False, time=5, task=None):
        img = self.get_image()
        cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(time)


if __name__=="__main__":
    env = Collector2DEnv()  # Task is sampled automatically

    obs = env.reset()
    done = False

    acc_reward = 0
    while not done:
        action = np.array([obs[0], obs[1]])
        obs, reward, done, task = env.step(action)
        print(obs)
        env.render(time=500)
        acc_reward += reward

    print("acc_reward:", acc_reward)