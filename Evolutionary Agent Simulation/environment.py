# environment.py

import numpy as np
from collections import OrderedDict
from config import SimulationConfig
from food import Food
from agent import Agent

class Environment:
    def __init__(self):
        self.agents = OrderedDict()
        self.foods = []
        self.tick_count = 0
        self.next_agent_id = 0
        self.main_agent_id = None
        self._initialize_environment()

    def _initialize_environment(self):
        for _ in range(SimulationConfig.INITIAL_FOOD_COUNT):
            self._spawn_food()
        for _ in range(SimulationConfig.NUM_AGENTS):
            self._spawn_agent()
        if self.agents:
            self.main_agent_id = next(iter(self.agents.keys()))

    def _spawn_food(self, position=None):
        if position is None:
            x = np.random.uniform(SimulationConfig.FOOD_RADIUS, SimulationConfig.PLANE_WIDTH - SimulationConfig.FOOD_RADIUS)
            y = np.random.uniform(SimulationConfig.FOOD_RADIUS, SimulationConfig.PLANE_HEIGHT - SimulationConfig.FOOD_RADIUS)
        else:
            x, y = position
        self.foods.append(Food(x, y, self.tick_count))

    def _spawn_agent(self):
        x = np.random.uniform(SimulationConfig.AGENT_RADIUS, SimulationConfig.PLANE_WIDTH - SimulationConfig.AGENT_RADIUS)
        y = np.random.uniform(SimulationConfig.AGENT_RADIUS, SimulationConfig.PLANE_HEIGHT - SimulationConfig.AGENT_RADIUS)
        agent_id = self.next_agent_id
        self.agents[agent_id] = Agent(agent_id, x, y)
        self.next_agent_id += 1
        return agent_id

    def step(self, actions_by_agent):
        self.tick_count += 1
        rewards = {agent_id: 0 for agent_id in self.agents if self.agents[agent_id].is_alive}
        
        # Process agent actions
        for agent_id, agent_actions in actions_by_agent.items():
            agent = self.agents.get(agent_id)
            if not agent or not agent.is_alive:
                continue

            agent_decision = agent.act(agent_actions)

            # --- NEW: Boundary collision and response ---
            if not (agent.radius <= agent.position[0] <= SimulationConfig.PLANE_WIDTH - agent.radius):
                agent.position[0] = np.clip(agent.position[0], agent.radius, SimulationConfig.PLANE_WIDTH - agent.radius)
                agent.take_damage(SimulationConfig.BOUNDARY_DAMAGE)
                rewards[agent_id] -= SimulationConfig.BOUNDARY_DAMAGE
            if not (agent.radius <= agent.position[1] <= SimulationConfig.PLANE_HEIGHT - agent.radius):
                agent.position[1] = np.clip(agent.position[1], agent.radius, SimulationConfig.PLANE_HEIGHT - agent.radius)
                agent.take_damage(SimulationConfig.BOUNDARY_DAMAGE)
                rewards[agent_id] -= SimulationConfig.BOUNDARY_DAMAGE

            if agent_decision["eat"]:
                food_eaten_index = None
                for i, food in enumerate(self.foods):
                    if np.linalg.norm(agent.position - food.position) < agent.radius + food.radius:
                        agent.hp += food.hp_value
                        agent.hp -= SimulationConfig.EAT_HP_COST
                        rewards[agent_id] += food.hp_value
                        rewards[agent_id] -= SimulationConfig.EAT_HP_COST
                        food_eaten_index = i
                        break
                if food_eaten_index is not None:
                    del self.foods[food_eaten_index]

            if agent_decision["attack"]:
                agent.hp -= SimulationConfig.ATTACK_HP_COST
                rewards[agent_id] -= SimulationConfig.ATTACK_HP_COST
                for other_agent_id, other_agent in self.agents.items():
                    if agent_id == other_agent_id or not other_agent.is_alive:
                        continue
                    if np.linalg.norm(agent.position - other_agent.position) < SimulationConfig.ATTACK_RANGE:
                        other_agent.take_damage(SimulationConfig.ATTACK_DAMAGE)
                        rewards[agent_id] += SimulationConfig.ATTACK_DAMAGE
                        rewards[other_agent_id] -= SimulationConfig.ATTACK_DAMAGE
        
        self.foods = [food for food in self.foods if self.tick_count - food.spawn_tick < SimulationConfig.FOOD_LIFETIME]

        dead_agents_ids = [agent_id for agent_id, agent in self.agents.items() if not agent.is_alive]
        for agent_id in dead_agents_ids:
            agent = self.agents[agent_id]
            for _ in range(SimulationConfig.DEATH_FOOD_DROP_AMOUNT):
                scatter_angle = np.random.uniform(0, 2 * np.pi)
                scatter_dist = np.sqrt(np.random.uniform(0, 1)) * SimulationConfig.DEATH_FOOD_DROP_SCATTER_RADIUS
                drop_x = agent.position[0] + scatter_dist * np.cos(scatter_angle)
                drop_y = agent.position[1] + scatter_dist * np.sin(scatter_angle)
                self._spawn_food(position=np.array([drop_x, drop_y]))
            
            del self.agents[agent_id]
            new_agent_id = self._spawn_agent()
            if agent_id == self.main_agent_id:
                self.main_agent_id = new_agent_id if new_agent_id is not None else next(iter(self.agents.keys()), None)

        if np.random.rand() < SimulationConfig.FOOD_SPAWN_RATE:
            self._spawn_food()

        observations = {agent_id: agent.get_sight_input(self) for agent_id, agent in self.agents.items() if agent.is_alive}
        hp_states = {agent_id: agent.hp for agent_id, agent in self.agents.items() if agent.is_alive}
        
        # The 'done' flag is not really used in a continuous simulation, so it's kept as False.
        return observations, rewards, False, {}, hp_states

    def reset(self):
        self.agents = OrderedDict()
        self.foods = []
        self.tick_count = 0
        self.next_agent_id = 0
        self.main_agent_id = None
        self._initialize_environment()
        observations = {agent_id: agent.get_sight_input(self) for agent_id, agent in self.agents.items() if agent.is_alive}
        hp_states = {agent_id: agent.hp for agent_id, agent in self.agents.items() if agent.is_alive}
        return observations, hp_states

    def get_drawable_entities(self):
        entities = []
        for food in self.foods:
            entities.append({"type": "food", "position": food.position, "radius": food.radius, "color": food.get_color()})
        for agent_id, agent in self.agents.items():
            if agent.is_alive:
                entities.append({"type": "agent", "position": agent.position, "orientation": agent.orientation, "radius": agent.radius, "color": agent.get_color(), "id": agent_id})
        return entities

    def get_main_agent_data(self):
        agent = self.agents.get(self.main_agent_id)
        if agent:
            rays = []
            for ray_direction in agent.get_sight_rays():
                start_point = agent.position
                end_point = agent.position + ray_direction * SimulationConfig.MAX_SIGHT_DISTANCE
                rays.append((start_point, end_point))
            
            return {
                "id": agent.id, "position": agent.position, "orientation": agent.orientation,
                "hp": agent.hp, "rays": rays, "sight_input": agent.get_sight_input(self)
            }
        return None