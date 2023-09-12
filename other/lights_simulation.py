import pygame
import numpy as np
import time


pygame.init()

class Lights:
    def __init__(self, x, y, red=False):
        self.x = x
        self.y = y
        if red:
            self.red = True
        else:
            self.red = False
    def change(self):
        self.red = not self.red

class Settings:
    def __init__(self):
        self.speed = 2

class Car:
    def __init__(self, settings, left=True):
        self.settings = settings
        self.cost = 0
        self.left = left
        if left:
            self.x = 2000
            self.y = 960
        else:
            self.x = 960
            self.y = 2000
        self.moving = True
    def move(self):
        if self.left:
            self.x -= self.settings.speed
        else:
            self.y -= self.settings.speed

class Simulation(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((2000, 2000))
        self.screen_rect = self.screen.get_rect()
        self.clock = pygame.time.Clock()
        self.running = True

        self.settings = Settings()
        # self.car = Car(self.settings, True)

        self.h_cars = []
        self.v_cars = []
        # self.h_cars.append(self.car)

        self.h_lights = Lights(1020, 900)
        self.v_lights = Lights(900, 1020, True)

        self.line = 100
        self.change_penalty = 500

        self.start_time = time.time()
        self.v_waiting = False
        self.h_waiting = False
        self.change_time = 2

        self.h_start_time = time.time()
        self.v_start_time = time.time()
        # self.font = pygame.font.SysFont(self.settings.font, self.settings.font_size)
        # self.font1 = pygame.font.SysFont(self.settings.font, 30)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.h_lights.change()
                        self.v_lights.change()
    
            h = np.random.binomial(1, 0.018)
            if h == 1:
                self.h_cars.append(Car(self.settings, True))

            v = np.random.binomial(1, 0.009)
            if v == 1:
                self.v_cars.append(Car(self.settings, False))
            
            self.screen.fill((0,0,0))
            try:
                if self.h_lights.red:
                    for car in self.h_cars:
                        if car.x >= self.h_lights.x + 2 and car.x <= self.h_lights.x + 5:
                            car.moving = False
                else:
                    for car in self.h_cars:
                        car.moving = True
            except:
                pass

            try:
                if self.v_lights.red:
                    for car in self.v_cars:
                        if car.y >= self.v_lights.y + 2 and car.y <= self.v_lights.y + 5:
                            car.moving = False
                else:
                    for car in self.v_cars:
                        car.moving = True
            except:
                pass

            h_cost = 0
            for car in self.h_cars:
                if car.x >= self.h_lights.x and car.x <= self.h_lights.x + self.line:
                    car.cost += 15
                    h_cost += car.cost 
                else:
                    car.cost = 0
                for another_car in self.h_cars:
                    if another_car.x <= car.x - 12 and another_car.x >= car.x - 15 and another_car.moving == False:
                        car.moving = False
                pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(car.x, car.y, 10, 10))
                if car.moving:
                    car.move()
                if car.x <= 0:
                    self.h_cars.remove(car)

            v_cost = 0
            for car in self.v_cars:
                if car.y >= self.v_lights.y and car.y <= self.v_lights.y + self.line + 30:
                    car.cost += 10
                    v_cost += car.cost 
                else:
                    car.cost = 0
                for another_car in self.v_cars:
                    if another_car.y <= car.y - 12 and another_car.y >= car.y - 15 and another_car.moving == False:
                        car.moving = False
                pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(car.x, car.y, 10, 10))
                if car.moving:
                    car.move()
                try:
                    if car.y <= 0:
                        self.h_cars.remove(car)
                except:
                    pass

            self.h_time_passed = time.time() - self.h_start_time
            self.v_time_passed = time.time() - self.v_start_time

            if h_cost > v_cost + self.change_penalty and self.h_lights.red and not (self.h_waiting or self.v_waiting):
                self.v_lights.red = True
                self.h_start_time = time.time()
                self.h_waiting = True
            elif v_cost > h_cost + self.change_penalty and self.v_lights.red and not (self.h_waiting or self.v_waiting):
                self.h_lights.red = True
                self.v_start_time = time.time()
                self.v_waiting = True
            if self.v_waiting and self.v_time_passed > self.change_time:
                self.v_lights.red = False
                self.v_waiting = False
            if self.h_waiting and self.h_time_passed > self.change_time:
                self.h_lights.red = False
                self.h_waiting = False
            print(f'h cost: {h_cost}')
            print(f'v cost: {v_cost}')

            
            pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(self.v_lights.x - 5, self.v_lights.y - 5, 30, 60))
            pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(self.h_lights.x - 5, self.h_lights.y - 5, 30, 60))

            if self.h_lights.red:
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.h_lights.x, self.h_lights.y, 20, 20))
                pygame.draw.rect(self.screen,  (50, 50, 50), pygame.Rect(self.h_lights.x, self.h_lights.y + 30, 20, 20))
            else:
                pygame.draw.rect(self.screen,  (50, 50, 50), pygame.Rect(self.h_lights.x, self.h_lights.y, 20, 20))
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.h_lights.x, self.h_lights.y + 30, 20, 20))

            if self.v_lights.red:
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.v_lights.x, self.v_lights.y, 20, 20))
                pygame.draw.rect(self.screen, (50, 50, 50), pygame.Rect(self.v_lights.x, self.v_lights.y + 30, 20, 20))
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), pygame.Rect(self.v_lights.x, self.v_lights.y, 20, 20))
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.v_lights.x, self.v_lights.y + 30, 20, 20))
            

            

            pygame.display.flip()
            self.clock.tick(60)
            
if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
    pygame.quit()
                