# visualization.py

import pygame
import numpy as np
import math
from config import SimulationConfig

class SimulationRenderer:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Evolutionary Simulation")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 24) # Font for text display

    def draw(self, entities, tick_count, main_agent_data=None, 
             recent_agent_rewards_at_death=None, recent_agent_lifespans=None): # New parameters
        self.screen.fill((0, 0, 0)) # Black background

        for entity in entities:
            if entity["type"] == "food":
                color = (int(entity["color"][0]*255), int(entity["color"][1]*255), int(entity["color"][2]*255))
                pygame.draw.circle(self.screen, color, (int(entity["position"][0]), int(entity["position"][1])), int(entity["radius"]))
            elif entity["type"] == "agent":
                color = (int(entity["color"][0]*255), int(entity["color"][1]*255), int(entity["color"][2]*255))
                pygame.draw.circle(self.screen, color, (int(entity["position"][0]), int(entity["position"][1])), int(entity["radius"]))
                
                # Draw agent orientation (a line from center to edge)
                end_x = entity["position"][0] + entity["radius"] * math.cos(math.radians(entity["orientation"]))
                end_y = entity["position"][1] + entity["radius"] * math.sin(math.radians(entity["orientation"]))
                pygame.draw.line(self.screen, (255, 255, 255), (int(entity["position"][0]), int(entity["position"][1])), (int(end_x), int(end_y)), 2)

                # Highlight main agent
                if main_agent_data and entity["id"] == main_agent_data["id"]:
                    pygame.draw.circle(self.screen, (255, 255, 0), (int(entity["position"][0]), int(entity["position"][1])), int(entity["radius"] + 2), 2) # Yellow border

        # Display tick count
        tick_text = self.font.render(f"Tick: {tick_count}", True, (255, 255, 255))
        self.screen.blit(tick_text, (10, 10))

        if main_agent_data:
            # Display Main Agent Info
            info_y = 40
            main_agent_id_text = self.font.render(f"Main Agent ID: {main_agent_data['id']}", True, (255, 255, 255))
            self.screen.blit(main_agent_id_text, (10, info_y))
            info_y += 20

            main_agent_hp_text = self.font.render(f"Main Agent HP: {main_agent_data['hp']:.1f}", True, (255, 255, 255))
            self.screen.blit(main_agent_hp_text, (10, info_y))
            info_y += 20

            # Display current cumulative reward
            if "current_cumulative_reward" in main_agent_data:
                current_reward_text = self.font.render(f"Current Reward: {main_agent_data['current_cumulative_reward']:.2f}", True, (255, 255, 255))
                self.screen.blit(current_reward_text, (10, info_y))
                info_y += 20

            # Display Sight Input HP (last element)
            # FIX: Use len() > 0 to check if the array is not empty
            if len(main_agent_data['sight_input']) > 0: # Corrected line
                sight_hp_text = self.font.render(f"Sight Input HP (last element): {main_agent_data['sight_input'][-1]:.2f}", True, (255, 255, 255))
                self.screen.blit(sight_hp_text, (10, info_y))
                info_y += 20

            # Display sampled actions
            if main_agent_data['action_outputs']:
                actions_text = self.font.render(
                    f"Actions: TC={main_agent_data['action_outputs'][0]:.0f}, TCC={main_agent_data['action_outputs'][1]:.0f}, TS={main_agent_data['action_outputs'][2]:.2f}, MF={main_agent_data['action_outputs'][3]:.0f}, MS={main_agent_data['action_outputs'][4]:.2f}, Eat={main_agent_data['action_outputs'][5]:.0f}, Attack={main_agent_data['action_outputs'][6]:.0f}",
                    True, (255, 255, 255)
                )
                self.screen.blit(actions_text, (10, info_y))
                info_y += 20


            # Draw sight rays (simplified for visualization)
            for start, end in main_agent_data["rays"]:
                pygame.draw.line(self.screen, (100, 100, 100), (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 1)

        # Display recent agent death statistics
        if recent_agent_rewards_at_death and recent_agent_lifespans:
            avg_reward = sum(recent_agent_rewards_at_death) / len(recent_agent_rewards_at_death)
            avg_lifespan = sum(recent_agent_lifespans) / len(recent_agent_lifespans)

            avg_reward_text = self.font.render(f"Avg. Reward (last {len(recent_agent_rewards_at_death)} deaths): {avg_reward:.2f}", True, (255, 255, 255))
            self.screen.blit(avg_reward_text, (self.width - avg_reward_text.get_width() - 10, 10))

            avg_lifespan_text = self.font.render(f"Avg. Lifespan (last {len(recent_agent_lifespans)} deaths): {avg_lifespan:.0f} ticks", True, (255, 255, 255))
            self.screen.blit(avg_lifespan_text, (self.width - avg_lifespan_text.get_width() - 10, 40))


        pygame.display.flip()

    def tick(self):
        self.clock.tick(SimulationConfig.RENDER_FPS)

    def quit(self):
        pygame.quit()