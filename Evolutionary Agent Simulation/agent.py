# agent.py

import numpy as np
from config import SimulationConfig

def ray_circle_intersection(ray_origin, ray_direction, circle_center, circle_radius):
    """Calculates the intersection point of a ray and a circle."""
    oc = ray_origin - circle_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - circle_radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # No intersection
    else:
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t1 >= 0: return t1
        if t2 >= 0: return t2
        return None

def ray_line_segment_intersection(ray_origin, ray_direction, p1, p2):
    """Calculates the intersection point of a ray and a line segment."""
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    dot_v2_v3 = np.dot(v2, v3)
    if np.abs(dot_v2_v3) < 1e-8:
        return None # Parallel lines
    t1 = np.cross(v2, v1) / dot_v2_v3
    t2 = np.dot(v1, v3) / dot_v2_v3
    if t1 >= 0.0 and (0.0 <= t2 <= 1.0):
        return t1
    return None

class Agent:
    def __init__(self, agent_id, x, y):
        self.id = agent_id
        self.position = np.array([x, y], dtype=float)
        self.hp = SimulationConfig.INITIAL_HP
        self.orientation = np.random.uniform(0, 360) # Degrees
        self.movement_speed = SimulationConfig.BASE_MOVEMENT_SPEED
        self.rotation_speed = SimulationConfig.BASE_ROTATION_SPEED
        self.is_alive = True
        self.radius = SimulationConfig.AGENT_RADIUS
        self.rnn_hidden_state = None

    def get_sight_rays(self):
        """Generates the directions of the sight rays."""
        rays = []
        angle_step = SimulationConfig.SIGHT_ANGLE / (SimulationConfig.NUM_SIGHT_RAYS - 1) if SimulationConfig.NUM_SIGHT_RAYS > 1 else 0
        start_angle = self.orientation - (SimulationConfig.SIGHT_ANGLE / 2)
        for i in range(SimulationConfig.NUM_SIGHT_RAYS):
            angle = np.deg2rad(start_angle + i * angle_step)
            direction = np.array([np.cos(angle), np.sin(angle)])
            rays.append(direction)
        return rays

    def get_sight_input(self, environment):
        """Generates agent's sensory input using precise ray-casting."""
        sight_inputs = []
        rays = self.get_sight_rays()
        
        # Define environment boundaries
        w, h = SimulationConfig.PLANE_WIDTH, SimulationConfig.PLANE_HEIGHT
        boundaries = [
            (np.array([0, 0]), np.array([w, 0])),
            (np.array([w, 0]), np.array([w, h])),
            (np.array([w, h]), np.array([0, h])),
            (np.array([0, h]), np.array([0, 0]))
        ]

        for ray_dir in rays:
            closest_dist = SimulationConfig.MAX_SIGHT_DISTANCE
            hit_entity_color = SimulationConfig.COLOR_EMPTY

            # Check for intersection with food
            for food in environment.foods:
                dist = ray_circle_intersection(self.position, ray_dir, food.position, food.radius)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    hit_entity_color = SimulationConfig.COLOR_FOOD
            
            # Check for intersection with other agents
            for other_agent in environment.agents.values():
                if other_agent.id == self.id or not other_agent.is_alive:
                    continue
                dist = ray_circle_intersection(self.position, ray_dir, other_agent.position, other_agent.radius)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    hit_entity_color = SimulationConfig.COLOR_AGENT

            # Check for intersection with boundaries
            for p1, p2 in boundaries:
                dist = ray_line_segment_intersection(self.position, ray_dir, p1, p2)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    hit_entity_color = SimulationConfig.COLOR_BOUNDARY

            sight_inputs.extend(hit_entity_color)

        sight_inputs.append(self.hp / SimulationConfig.INITIAL_HP) # Normalize HP
        return np.array(sight_inputs, dtype=np.float32)

    def _apply_movement(self, move_forward_value):
        if move_forward_value > 0:
            move_distance = self.movement_speed
            rad_orientation = np.deg2rad(self.orientation)
            self.position[0] += move_distance * np.cos(rad_orientation)
            self.position[1] += move_distance * np.sin(rad_orientation)
            self.hp -= max(0, self.movement_speed - SimulationConfig.BASE_MOVEMENT_SPEED) * SimulationConfig.SPEED_HP_COST_FACTOR

    def _apply_rotation(self, turn_direction): # -1 for counter-clockwise, 1 for clockwise
        self.orientation = (self.orientation + turn_direction * self.rotation_speed) % 360
        self.hp -= max(0, self.rotation_speed - SimulationConfig.BASE_ROTATION_SPEED) * SimulationConfig.SPEED_HP_COST_FACTOR

    def take_damage(self, damage):
        self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False

    def act(self, action_outputs):
        turn_clockwise_prob = action_outputs[0]
        turn_counterclockwise_prob = action_outputs[1]
        self.rotation_speed = SimulationConfig.BASE_ROTATION_SPEED + action_outputs[2] * (SimulationConfig.MAX_ROTATION_SPEED - SimulationConfig.BASE_ROTATION_SPEED)
        move_forward_prob = action_outputs[3]
        self.movement_speed = SimulationConfig.BASE_MOVEMENT_SPEED + action_outputs[4] * (SimulationConfig.MAX_MOVEMENT_SPEED - SimulationConfig.BASE_MOVEMENT_SPEED)
        eat_prob = action_outputs[5]
        attack_prob = action_outputs[6]

        # Simplified and clearer rotation logic
        if turn_clockwise_prob > 0.5 and turn_clockwise_prob > turn_counterclockwise_prob:
            self._apply_rotation(1) # Clockwise
        elif turn_counterclockwise_prob > 0.5:
            self._apply_rotation(-1) # Counter-clockwise

        if move_forward_prob > 0.5:
            self._apply_movement(1)

        self.hp -= SimulationConfig.HP_DECAY_RATE
        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False

        return {
            "eat": eat_prob > 0.5,
            "attack": attack_prob > 0.5
        }

    def get_state(self):
        return self.hp, self.position, self.orientation, self.movement_speed, self.rotation_speed, self.is_alive

    def get_color(self):
        return SimulationConfig.COLOR_AGENT