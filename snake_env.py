import pygame
import random
import numpy as np

CELL_SIZE = 20
GRID_SIZE = 40
WINDOW_SIZE = 800  # 20 * 40 = 800

STRAIGHT = 0
RIGHT = 1
LEFT = 2
ACTIONS = [STRAIGHT, RIGHT, LEFT]

UP = (0, -1)
DOWN = (0, 1)
LEFT_DIR = (-1, 0)
RIGHT_DIR = (1, 0)
DIRECTIONS = [RIGHT_DIR, DOWN, LEFT_DIR, UP]

class SnakeEnv:
    def __init__(self, render=False):
        self.render_mode = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = RIGHT_DIR
        self.head = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = [self.head]
        self.score = 0
        self.frame = 0
        self.food = self._place_food()
        return self._get_state()

    def step(self, action):
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False

        if self.head == self.food:
            reward = 10
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        if self._is_collision():
            reward = -10
            game_over = True
            return self._get_state(), reward, game_over, self.score

        if self.render_mode:
            self._draw()

        return self._get_state(), reward, game_over, self.score

    def _place_food(self):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in self.snake:
                return pos

    def _move(self, action):
        idx = DIRECTIONS.index(self.direction)
        if action == LEFT:
            idx = (idx - 1) % 4
        elif action == RIGHT:
            idx = (idx + 1) % 4
        self.direction = DIRECTIONS[idx]
        x, y = self.snake[0]
        dx, dy = self.direction
        self.head = (x + dx, y + dy)

    def _is_collision(self):
        x, y = self.head
        return (
            x < 0 or x >= GRID_SIZE or
            y < 0 or y >= GRID_SIZE or
            self.head in self.snake[1:]
        )

    def _next_position(self, action):
        idx = DIRECTIONS.index(self.direction)
        if action == LEFT:
            idx = (idx - 1) % 4
        elif action == RIGHT:
            idx = (idx + 1) % 4
        dir = DIRECTIONS[idx]
        x, y = self.snake[0]
        return (x + dir[0], y + dir[1])

    def _collision(self, point):
        x, y = point
        return (
            x < 0 or x >= GRID_SIZE or
            y < 0 or y >= GRID_SIZE or
            point in self.snake
        )

    def _get_state(self):
        head = self.snake[0]

        danger_straight = self._collision(self._next_position(STRAIGHT))
        danger_left = self._collision(self._next_position(LEFT))
        danger_right = self._collision(self._next_position(RIGHT))

        dir_l = self.direction == LEFT_DIR
        dir_r = self.direction == RIGHT_DIR
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        apple_dx = self.food[0] - head[0]
        apple_dy = self.food[1] - head[1]
        manhattan_dist = abs(apple_dx) + abs(apple_dy)

        apple_in_view = int(
            (dir_l and apple_dx < 0) or
            (dir_r and apple_dx > 0) or
            (dir_u and apple_dy < 0) or
            (dir_d and apple_dy > 0)
        )

        state = [
            int(danger_left),
            int(danger_straight),
            int(danger_right),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            apple_dx / GRID_SIZE,
            apple_dy / GRID_SIZE,
            manhattan_dist / (GRID_SIZE * 2),
            apple_in_view,
            len(self.snake) / (GRID_SIZE * GRID_SIZE)
        ]
        return np.array(state, dtype=np.float32)

    def _draw(self):
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake Environment")

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for pos in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS für flüssige Darstellung


    def close(self):
        if self.render_mode:
            pygame.quit()
