# food.py

import numpy as np
from config import SimulationConfig

class Food:
    def __init__(self, x, y, spawn_tick=0):
        self.position = np.array([x, y], dtype=float)
        self.hp_value = SimulationConfig.FOOD_HP_GAIN
        self.radius = SimulationConfig.FOOD_RADIUS
        self.spawn_tick = spawn_tick # Track when the food was spawned

    def get_color(self):
        return SimulationConfig.COLOR_FOOD