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
    def __init__(self, settings):
        self.settings = settings
        
    def generate_position(self, snake_positions):
        self.position = [self.settings.snake_size*random.randint(0, self.settings.board_width-1), self.settings.snake_size*random.randint(0, self.settings.board_width-1)]
        while self.position in snake_positions:
          self.position = [self.settings.snake_size*random.randint(0, self.settings.board_width-1), self.settings.snake_size*random.randint(0, self.settings.board_width-1)]

    def show(self, screen):
        pygame.draw.rect(screen, (100, 100, 100), pygame.Rect(self.position[0], self.position[1], self.settings.snake_size, self.settings.snake_size))


class Snake:
    def __init__(self, settings):
        self.settings = settings

        self.positions = [[90, 90], [90+self.settings.snake_size, 90], [90+2*self.settings.snake_size, 90]]
        self.head = self.positions[0].copy()

    
    def show(self, screen):
        for position in self.positions[1:]:
            pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(position[0], position[1], self.settings.snake_size, self.settings.snake_size))
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(self.positions[0][0], self.positions[0][1], self.settings.snake_size, self.settings.snake_size))
    
    def check_collision(self):
        if self.positions[0] in self.positions[1:]:
            return True
        if self.positions[0][0] < 0 or self.positions[0][0] > self.settings.screen_width-self.settings.snake_size:
            return True
        if self.positions[0][1] < 0 or self.positions[0][1] > self.settings.screen_height-self.settings.snake_size:
            return True
        return False
    
    def move(self, direction: str):
        if direction == 'up':
            self.head[1] -= self.settings.snake_size
        elif direction == 'down':
            self.head[1] += self.settings.snake_size
        elif direction == 'left':
            self.head[0] -= self.settings.snake_size
        elif direction == 'right':
            self.head[0] += self.settings.snake_size
        self.positions.insert(0, self.head.copy())
        self.positions.pop()
    
    def collect_apple(self, apple_position):
        self.positions.insert(0, apple_position.copy())


class Settings:
    def __init__(self):
        self.fps = 10
        self.snake_size = 30
        self.board_width, self.board_height = 6, 6

        self.screen_width = self.snake_size*self.board_width
        self.screen_height = self.snake_size*self.board_height


class GameEnv(gym.Env):
    def __init__(self, train=True):
        super(GameEnv, self).__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=-1, high=1000, shape=(3*2+4+2,), dtype=np.int32)

        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        self.screen_rect = self.screen.get_rect()

        self.snake = Snake(self.settings)

        self.apple = Apple(self.settings)

        self.playing = True
        self.prev_action = 'left'

        self.stop = False

        self.clock = pygame.time.Clock()

        self.lose_penalty = -10
        self.apple_reward = 1

        self.show_after = 20_000
        self.show_every = 1_000
        self.epochs = 0
        self.steps = 0
        self.apples_collected = 0

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
            self.apple = Apple(self.settings)
            self.apple.generate_position(self.snake.positions)
            self.apples_collected += 1
            reward += self.apple_reward
        self.snake.move(action)

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
        if self.epochs > self.show_after or self.epochs % self.show_every == 0:
            self.train = False
        else:
            self.train = True

        if self.steps > 1000:
            done = True
            reward = self.lose_penalty
        
        self.prev_action = action

        self.steps += 1

        # set observation
        observation = self.snake.positions[:3].copy()
        # observation += [[-1, -1]]*(4 - len(self.snake.positions)) if 4 - len(self.snake.positions) > 0
        observation = np.append(observation, self.apple.position[0])
        observation = np.append(observation, self.apple.position[1])
        observation = np.append(observation, self._to_indicator(self.prev_action))

        if done:
            print(f'epochs: {self.epochs}, reward: {reward}, steps: {self.steps}, len_sn_pos: {self.apples_collected}')

        return observation, reward, done, info

    def reset(self):
        self.apples_collected = 0
        self.steps = 0
        self.epochs += 1
        self.snake = Snake(self.settings)
        self.apple = Apple(self.settings)
        self.apple.generate_position(self.snake.positions)
        self.prev_action = 'left'

        observation = self.snake.positions[:3].copy()
        # observation += [[-1, -1]]*(SNAKE_LEN_GOAL - len(self.snake.positions))
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
    model = PPO.load('/home/jakub/python/reinforcement_learning/snake_rl/models/1702061310/980000', env, tensorboard_log=tensorboard_dir)


    TIMESTEPS = 20_000
    iters = 0
    # while not env.stop:
    for i in range(200):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False)
        model.save(f"{models_dir}/{TIMESTEPS*iters}")
        iters += 1


if __name__ == '__main__':
    main()
