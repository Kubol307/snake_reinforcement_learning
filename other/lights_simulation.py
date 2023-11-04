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

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(self.x - 5, self.y - 5, 30, 60))
        pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(self.x + 8, self.y + 50, 5, 40))
        if self.red:
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.x, self.y, 20, 20))
            pygame.draw.rect(screen,  (50, 50, 50), pygame.Rect(self.x, self.y + 30, 20, 20))
        else:
            pygame.draw.rect(screen,  (50, 50, 50), pygame.Rect(self.x, self.y, 20, 20))
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(self.x, self.y + 30, 20, 20))



class Settings:
    def __init__(self):
        self.speed = 3
        self.speed_upper = 2
        self.speed_lower = 4

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
        self.speed = np.random.uniform(self.settings.speed_lower, self.settings.speed_upper)

        self.width = 55
        self.height = 20

        self.image = pygame.image.load('car.png')
        self.image = pygame.transform.scale(self.image, (55, 20))

        if not self.left:
            self.image = pygame.transform.rotate(self.image, -90)
    def move(self):
        if self.left:
            self.x -= self.speed
        else:
            self.y -= self.speed
    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

class Simulation(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((2000, 2000))
        pygame.display.set_caption('Traffic simulator')
        self.screen_rect = self.screen.get_rect()
        self.clock = pygame.time.Clock()
        self.running = True

        self.settings = Settings()
        # self.car = Car(self.settings, True)

        self.h_cars = []
        self.v_cars = []
        # self.h_cars.append(self.car)

        self.h_lights = Lights(1020, 850)
        self.v_lights = Lights(1020, 1150, True)

        self.line = 100
        self.distance = 150
        self.change_penalty = 10_000

        self.start_time = time.time()
        self.time_passed = 0
        self.v_changing = False
        self.h_changing = False
        self.change_time = 1

        self.acceleration = 0.05

        self.distance_before_starting = 20

        self.road_image = pygame.image.load('road.png')
        self.road_image = pygame.transform.scale(self.road_image, (1000, 1000))

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
    
            h = np.random.binomial(1, 0.01)
            if h == 1:
                car = Car(self.settings, True)
                if len(self.h_cars) > 0 and self.h_cars[-1].x < 2000 - car.width - 20:
                    self.h_cars.append(car)
                elif len(self.h_cars) == 0:
                    self.h_cars.append(car)
                

            v = np.random.binomial(1, 0.01)
            if v == 1:
                v_car = Car(self.settings, False)
                if len(self.v_cars) > 0 and self.h_cars[-1].y < 2000 - v_car.width - 20:
                    self.v_cars.append(v_car)
                elif len(self.v_cars) == 0:
                    self.v_cars.append(v_car)
            
            self.screen.fill((0,0,0))
            self.screen.blit(self.road_image, (420, 550))
            self.screen.blit(self.road_image, (420-850, 550))
            self.screen.blit(self.road_image, (420+850, 550))
            self.screen.blit(self.road_image, (420, 550-850))
            self.screen.blit(self.road_image, (420, 550+850))

            try:
                for i, car in enumerate(self.h_cars): 
                    if car.x >= self.h_lights.x + 2 and car.x <= self.h_lights.x + 5 and self.h_lights.red:
                        car.speed = 0
                        
                    elif i > 0 and self.h_cars[i - 1].x <= car.x - self.distance_before_starting:
                        if car.speed < self.settings.speed:
                            car.speed += self.acceleration
                    elif i == 0:
                        if car.speed < self.settings.speed:
                            car.speed += self.acceleration
            except:
                pass

            try:
                for i, car in enumerate(self.v_cars): 
                    if car.y >= self.v_lights.y + 2 and car.y <= self.v_lights.y + 5 and self.v_lights.red:
                        car.speed = 0
                        
                    elif i > 0 and self.v_cars[i - 1].y <= car.y - self.distance_before_starting:
                        if car.speed < self.settings.speed:
                            car.speed += self.acceleration
                    elif i == 0:
                        if car.speed < self.settings.speed:
                            car.speed += self.acceleration
            except:
                pass

            h_cost = 0
            for car in self.h_cars:
                if not self.v_changing and not self.h_changing:
                    if car.x >= self.h_lights.x and car.x <= self.h_lights.x + self.distance:
                        car.cost += 15
                        h_cost += car.cost 
                    elif car.x >= self.h_lights.x and car.x <= self.h_lights.x + self.line:
                        car.cost += 50
                        h_cost += car.cost 
                    else:
                        car.cost = 0

                for another_car in self.h_cars:
                    if another_car.x <= car.x - 40 and another_car.x >= car.x - 55 and another_car.speed < car.speed:
                        car.speed = another_car.speed
                    elif another_car.x <= car.x - 55 and another_car.x >= car.x - 140 and another_car.speed < car.speed:
                        if car.speed > 1:
                            car.speed -= 0.1

                    # if car.speed < self.settings.speed:
                    #     car.speed += self.acceleration


                # pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(car.x, car.y, 10, 10))
                car.draw(self.screen)
                if car.moving:
                    car.move()
                if car.x <= 0:
                    self.h_cars.remove(car)

            v_cost = 0
            for car in self.v_cars:
                if not self.v_changing and not self.h_changing:
                    if car.y >= self.v_lights.y and car.y <= self.v_lights.y + self.distance:
                        car.cost += 15
                        v_cost += car.cost 
                    elif car.y >= self.v_lights.y and car.y <= self.v_lights.y + self.line:
                        car.cost += 50
                        v_cost += car.cost 
                    else:
                        car.cost = 0

                for another_car in self.v_cars:
                    if another_car.y <= car.y - 40 and another_car.y >= car.y - 55 and another_car.speed < car.speed:
                        car.speed = another_car.speed
                    elif another_car.y <= car.y - 55 and another_car.y >= car.y - 140 and another_car.speed < car.speed:
                        if car.speed > 1:
                            car.speed -= 0.1

                # pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(car.x, car.y, 10, 10))
                car.draw(self.screen)
                if car.moving:
                    car.move()
                try:
                    if car.y <= 0:
                        self.v_cars.remove(car)
                except:
                    pass

            if self.v_changing or self.h_changing:
                self.time_passed = time.time() - self.start_time


            if not self.v_changing and not self.h_changing:
                # if h_cost == 0 and v_cost > 0 and self.v_lights.red:
                #     self.h_lights.change()
                #     self.h_lights.draw(self.screen)
                #     self.start_time = time.time()
                #     self.v_changing = True
                #     print('both red')
                # elif v_cost == 0 and h_cost > 0 and self.h_lights.red:
                #     self.v_lights.change()
                #     self.v_lights.draw(self.screen)
                #     self.start_time = time.time()
                #     self.h_changing = True
                #     print('both red')
                if h_cost > v_cost + self.change_penalty and self.h_lights.red:
                    self.v_lights.change()
                    self.v_lights.draw(self.screen)
                    self.start_time = time.time()
                    self.h_changing = True
                    print('both red')

                elif v_cost > h_cost + self.change_penalty and self.v_lights.red:
                    self.h_lights.change()
                    self.h_lights.draw(self.screen)
                    self.start_time = time.time()
                    self.v_changing = True
                    print('both red')

            if self.v_changing and self.time_passed > self.change_time:
                self.v_lights.change()
                self.v_changing = False
                self.time_passed = 0
                print('v green')
            if self.h_changing and self.time_passed > self.change_time:
                self.h_lights.change()
                print('h green')
                self.time_passed = 0
                self.h_changing = False

            self.v_lights.draw(self.screen)
            self.h_lights.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)
            
if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
    pygame.quit()
                