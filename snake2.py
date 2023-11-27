import gym
from gym import spaces
import pygame
import numpy as np
import random
from stable_baselines3 import PPO
import os
import time



SNAKE_LEN_GOAL = 30

class Apple:
    def __init__(self):
        self.position = [30*random.randint(0, 32), 30*random.randint(0, 32)]

    def show(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(self.position[0], self.position[1], 30, 30))


class Snake:
    def __init__(self):
        self.positions = [[180, 180], [210, 180], [240, 180]]
        self.head = self.positions[0].copy()

    def show(self, screen):
        for position in self.positions[1:]:
            pygame.draw.rect(screen, (0,255,0), pygame.Rect(position[0], position[1], 30, 30))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.positions[0][0], self.positions[0][1], 30, 30))

    def check_collision(self):
        if self.positions[0] in self.positions[1:]:
            return True
        if self.positions[0][0] < 0 or self.positions[0][0] > 990:
            return True
        if self.positions[0][1] < 0 or self.positions[0][1] > 990:
            return True
        return False

    def move(self, direction: str):
        if direction == 'up':
            self.head[1] -= 30
        elif direction == 'down':
            self.head[1] += 30
        elif direction == 'left':
            self.head[0] -= 30
        elif direction == 'right':
            self.head[0] += 30
        self.positions.insert(0, self.head.copy())
        self.positions.pop()

    def collect_apple(self, apple_position):
        self.positions.insert(0, apple_position.copy())



class Settings:
    def __init__(self):
        self.fps = 10
        self.screen_width = 1000
        self.screen_height = 1000


class GameEnv(gym.Env):
    def __init__(self, train=True):
        super(GameEnv, self).__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=-10_000, high=1000, shape=(SNAKE_LEN_GOAL*2+4+2,), dtype=np.int32)

        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        self.screen_rect = self.screen.get_rect()

        self.snake = Snake()

        self.apple = Apple()

        self.playing = True
        self.prev_action = 'left'

        self.stop = False

        self.clock = pygame.time.Clock()

        self.lose_penalty = -10_000
        self.apple_reward = 5_000

        self.show_after = 10_000
        self.epochs = 0
        self.steps = 0

        self.train = True
    def step(self, action):
        reward = 0
        done = False
        info = {}

        # enable quiting by closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.stop = True

        if action == 0 and self.prev_action != 'right':
            action = 'left'
        elif action == 1 and self.prev_action != 'left':
            action = 'right'
        elif action == 2 and self.prev_action != 'down':
            action = 'up'
        elif action == 3 and self.prev_action != 'up':
            action = 'down'
        else:
            action = self.prev_action

        # check if snake ate apple
        if self.apple.position in self.snake.positions:
            self.snake.collect_apple(self.apple.position)
            del self.apple
            self.apple = Apple()
            reward += self.apple_reward
        self.snake.move(action)

        euclidean_dist_to_apple = 250 - np.linalg.norm(
            np.array(self.snake.positions) - np.array(self.apple.position))

        reward += euclidean_dist_to_apple*0.1

        # draw everything
        if not self.train:
            self.screen.fill((0, 0, 0))
            self.apple.show(self.screen)
            self.snake.show(self.screen)
            pygame.display.flip()
            self.clock.tick(self.settings.fps)

        if self.snake.check_collision():
            done = True
            reward = self.lose_penalty

        # wait for fps
        if self.epochs > self.show_after:
            self.train = False
        elif self.steps > 1000:
            done = True
            reward = self.lose_penalty

        self.prev_action = action

        self.steps += 1

        # set observation
        observation = self.snake.positions.copy()
        observation += [[-1, -1]]*(SNAKE_LEN_GOAL - len(self.snake.positions))
        observation = np.append(observation, self.apple.position[0])
        observation = np.append(observation, self.apple.position[1])
        observation = np.append(observation, self._to_indicator(self.prev_action))

        if done:
            print(f'epochs: {self.epochs}, reward: {reward}, steps: {self.steps}, len_sn_pos: {len(self.snake.positions)}')

        return observation, reward, done, info

    def reset(self):
        self.steps = 0
        self.epochs += 1
        self.snake = Snake()
        self.apple = Apple()
        self.prev_action = 'left'

        observation = self.snake.positions.copy()
        observation += [[-1, -1]]*(SNAKE_LEN_GOAL - len(self.snake.positions))
        observation = np.append(observation, self.apple.position[0])
        observation = np.append(observation, self.apple.position[1])
        observation = np.append(observation, self._to_indicator(self.prev_action))
        return observation

    def _to_indicator(self, action):
        if action == 'left':
            return [1, 0, 0, 0]
        elif action == 'right':
            return [0, 1, 0, 0]
        elif action == 'up':
            return [0, 0, 1, 0]
        elif action == 'down':
            return [0, 0, 0, 1]
        else:
            raise ValueError('Invalid action')

def main():
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"
    tensorboard_dir = f"tensorboard/{int(time.time())}/"

    # create dirs if not exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    env = GameEnv()
    env.reset()

    # model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001, tensorboard_log=tensorboard_dir)
    model = PPO.load('/home/jakub/python/reinforcement_learning/snake_rl/models/1701120994/0', env, tensorboard_log=tensorboard_dir)


    TIMESTEPS = 3_000_000
    iters = 0
    # while not env.stop:
    model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    iters += 1


if __name__ == '__main__':
    main()
