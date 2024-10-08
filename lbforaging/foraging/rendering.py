import pygame
import os

# Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_PLAYER = (200, 100, 0)
_RED = (255, 0, 0)

script_dir = os.path.dirname(__file__)

img_apple = pygame.image.load(os.path.join(script_dir, "icons/apple.png"))
img_banana = pygame.image.load(os.path.join(script_dir, "icons/banana.png"))
img_agent = pygame.image.load(os.path.join(script_dir, "icons/agent.png"))


class Viewer:
    def __init__(self, world_size):

        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20
        self.player_icon_size = 40

        self.name_font_size = 20
        self.level_font_size = 20
        pygame.init()
        self._screen = pygame.display.set_mode((self.cols * self.grid_size + 1, self.rows * self.grid_size + 1))
        self._name_font = pygame.font.SysFont("monospace", self.name_font_size, bold=True)
        self._level_font = pygame.font.SysFont("monospace", self.level_font_size)

        self.img_apple = pygame.transform.scale(img_apple, (self.icon_size, self.icon_size))
        self.img_banana = pygame.transform.scale(img_banana, (self.icon_size, self.icon_size))
        self.img_agent = pygame.transform.scale(img_agent, (self.player_icon_size, self.player_icon_size))

    def render(self, env):

        self._screen.fill(_BLACK)
        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _draw_grid(self):
        for r in range(self.rows + 1):
            pygame.draw.line(
                self._screen,
                _WHITE,
                (0, self.grid_size * r),
                (self.grid_size * self.cols, self.grid_size * r),
            )
        for c in range(self.cols + 1):
            pygame.draw.line(
                self._screen,
                _WHITE,
                (self.grid_size * c, 0),
                (self.grid_size * c, self.grid_size * self.rows),
            )

    def _draw_food(self, env):
        for r in range(self.rows):
            for c in range(self.cols):
                for food_type in env.food_types:
                    field = env.food_type_mapping[food_type]
                    if field[r, c] != 0:
                        if food_type == "apple":
                            self._draw_population_in_cell(self.img_apple,
                                                          (self.grid_size * c, self.grid_size * r),
                                                          field[r, c])
                        elif food_type == "banana":
                            self._draw_population_in_cell(self.img_banana,
                                                          (self.grid_size * c, self.grid_size * r),
                                                          field[r, c])
                        else:
                            raise ValueError(f"Unknown food type {food_type}")

    def _draw_population_in_cell(self, img, location, number):
        offset = 5
        coords = [(0, 0), (0, self.icon_size), (self.icon_size, 0), (self.icon_size, self.icon_size)]

        for i in coords[:number]:
            self._screen.blit(img, (location[0] + i[0] + offset, location[1] + i[1] + offset))

    def _draw_players(self, env):
        for idx, (player, status) in enumerate(zip(env.players, env.active_agents)):
            if not status:
                continue
            r, c = player.position
            self._draw_population_in_cell(self.img_agent, (self.grid_size * c, self.grid_size * r), 1)
            # self._screen.blit(self.img_agent, (self.grid_size * c + 5, self.grid_size * r + 5))
            self._screen.blit(
                self._name_font.render(f"L{player.level}", 1, _PLAYER),
                (self.grid_size * c + self.grid_size // 3 - 14, self.grid_size * r + self.grid_size // 3 + 15),
            )

            self._screen.blit(
                self._name_font.render(f"N{idx+1}", 1, _PLAYER),
                (self.grid_size * c + self.grid_size // 3 + 10, self.grid_size * r + self.grid_size // 3 + 15),
            )

    def save_image(self, path="screenshot.png"):
        pygame.image.save(self._screen, path)
