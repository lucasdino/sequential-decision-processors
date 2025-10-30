from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from transformers import AutoTokenizer

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

class GeneralEnvironmentManager(EnvironmentManagerBase):
    # Added config because it is annoying to have to go into the code to switch values.
    def __init__(self, envs, projection_f, env_name, istrain = True, config=None):
        self.buffers = None
        self.istrain = istrain
        self.print_counter = False
        self.ttp_switch = False
        self.config = config
        self.env_name = env_name
        self.tok = AutoTokenizer.from_pretrained(self.config['actor_rollout_ref']['model']['path'], use_fast=True)
        super().__init__(envs, projection_f, env_name)
    
    # Stripped down general manager to handle all of the frameworks from TALES
    def reset(self):
        obs, infos = self.envs.reset()

        if self.buffers is not None:
            self.buffers.clear()

        self.buffers = [[] for _ in range(len(obs))]
        self.tasks = []
        self.pre_text_obs = obs

        full_text_obs = self.build_text_obs(obs, init=True)
        self.print_counter = True
        return {'text': full_text_obs, 'image': None, 'anchor': obs.copy()}, infos
    
    def step(self, text_actions: List[str], refined_responses: List[str] = None):
        actions, valids, thinking = self.projection_f(self.config['env']['prompt_template'], text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
  
        # Get rid of negative rewards if using the zero-centered reward mode or goal only
        if self.config['env']['reward_mode'] == 'zero-centered':
            rewards = [max(0, r) for r in rewards]
        elif self.config['env']['reward_mode'] == 'goal-only':
            rewards = [r if r > 9 else 0 for r in rewards]
        elif self.config['env']['reward_mode'] == 'negative-test':
            if not self.ttp_switch:
                rewards = [r if r < 0 else 0 for r in rewards]
        else:
            rewards = rewards


        self.save_to_history_buffer(self.pre_text_obs, actions, thinking)
        self.pre_text_obs = next_obs

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': self.build_text_obs(next_obs), 
                             'image': None, 
                             'anchor': next_obs.copy()}
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
 

        return next_observations, rewards, dones, infos
        

    def build_text_obs(self, text_obs: List[str], init: bool = False, history_length: int = 100) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # Get last `history_length` steps
            recent_history = self.buffers[i][-history_length:]
            valid_history_length = len(recent_history)
            start_index = len(self.buffers[i]) - valid_history_length
            action_history = ""
            action_history_with_think_arr = []
            for j, record in enumerate(recent_history):
                step_number = start_index + j + 1
                action = record["action"]
                env_obs = record["text_obs"]
                if "thinking_trace" in record.keys():
                    thinking_trace = record["thinking_trace"]
                else:
                    thinking_trace = "None or invalid thinking trace."
                action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                action_history_with_think_arr.append(f"\n[Observation {step_number}: '{env_obs}', Thoughts {step_number}: '{thinking_trace}' Action {step_number}: '{action}']")
            action_history_with_think = "".join(action_history_with_think_arr)

            # Check the token length
            attempts = 0
            if self.tok is not None:
                token_length = len(self.tok.encode(action_history_with_think))
                # If the token length is too long, we need to truncate the history (1000 is just for an extra safety margin)
                while token_length >= (self.config['actor_rollout_ref']['actor']['ppo_max_token_len_per_gpu'] - 1000):
                    action_history_with_think_arr.pop(0)
                    action_history_with_think = "".join(action_history_with_think_arr)
                    token_length = len(self.tok.encode(action_history_with_think))
                    attempts += 1
                    # If attempts go above 4/5 of the max steps, just default to the regular action history.
                    if attempts > (self.config['env']['max_steps'] * 0.8):
                        print(f"Warning: Too many attempts to truncate the action history for {self.env_name}. Defaulting to regular action history.")
                        action_history_with_think = action_history
                        break

            if self.config['env']['prompt_template'] == "basecase":
                GENERAL_TEMPLATE = general_INST_FIRST
            elif self.config['env']['prompt_template'] == 'sctq_inst_first':   
                GENERAL_TEMPLATE = general_SCRTQ_INST_FIRST
            elif self.config['env']['prompt_template'] == 'inst_first_with_think':
                GENERAL_TEMPLATE = general_INST_FIRST_WITH_THINK
            elif self.config['env']['prompt_template'] == 'sctq_inst_first_with_think':
                GENERAL_TEMPLATE = general_SCRTQ_INST_FIRST_WITH_THINK  
            elif self.config['env']['prompt_template'] == 'sctq_inst_first_with_think_extended':
                GENERAL_TEMPLATE = general_SCRTQ_INST_FIRST_WITH_THINK_EXTENDED
            else:
                raise ValueError(f"Unknown prompt template: {self.config.env.prompt_template}")

            obs = GENERAL_TEMPLATE.format(
                task_description=[],
                step_count=len(self.buffers[i]),
                history_length=valid_history_length,
                action_history=action_history.strip(),
                action_history_with_think=action_history_with_think.strip(),
                current_step=len(self.buffers[i]) + 1,
                current_observation=text_obs[i],
                admissible_actions=""
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, thinking_trace):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'thinking_trace': thinking_trace[i]})

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

class LifeGateEnvironmentManager(EnvironmentManagerBase):
    # Added config because it is annoying to have to go into the code to switch values.
    def __init__(self, envs, projection_f, env_name, istrain = True, config=None):
        self.buffers = None
        self.istrain = istrain
        self.print_counter = False
        self.ttp_switch = False
        self.config = config
        self.tok = AutoTokenizer.from_pretrained(self.config['actor_rollout_ref']['model']['path'], use_fast=True)
        super().__init__(envs, projection_f, env_name)
    
    # Modified by Chris 6/25:
    # Basing this on a mix of alfworld and webshop environment managers (since their implementation of alfworld has the images and we do not)
    def reset(self):
        obs, infos = self.envs.reset()

        if self.buffers is not None:
            self.buffers.clear()

        self.buffers = [[] for _ in range(len(obs))]
        self.tasks = []
        self.extract_task(obs)
        self.pre_text_obs = obs

        full_text_obs = self.build_text_obs(obs, init=True)
        self.print_counter = True
        # Try also using the test-time prompt during training
        if self.ttp_switch:
            self.ttp_switch = False
        else:
            self.ttp_switch = True
        return {'text': full_text_obs, 'image': None, 'anchor': obs.copy()}, infos
    
    def step(self, text_actions: List[str], refined_responses: List[str] = None):
        actions, valids, thinking = self.projection_f(self.config['env']['prompt_template'], text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
  
        if self.config['env']['reward_mode'] == 'zero-centered':
            # If using the zero-centered reward mode, we do not want negative rewards
            rewards = [max(0, r) for r in rewards]
        elif self.config['env']['reward_mode'] == 'goal-only':
            # Get rid of the living reward too
            rewards = [r if r > 9 else 0 for r in rewards]
        elif self.config['env']['reward_mode'] == 'negative-test':
            # Only allow negative rewards when using the test-time prompt
            if not self.ttp_switch:
                rewards = [r if r < 0 else 0 for r in rewards]
        else:
            # Unmodified
            rewards = rewards

        print("Env manager refined responses:", refined_responses)

        self.save_to_history_buffer(self.pre_text_obs, actions, thinking)
        self.pre_text_obs = next_obs

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': self.build_text_obs(next_obs), 
                             'image': None, 
                             'anchor': next_obs.copy()}
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for _ in text_obs:
            self.tasks.append("You are in a gridworld style maze. There is a lifegate on the other side. Try to reach it before you die.")
        

    def build_text_obs(self, text_obs: List[str], init: bool = False, history_length: int = 100) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # Get last `history_length` steps
            recent_history = self.buffers[i][-history_length:]
            valid_history_length = len(recent_history)
            start_index = len(self.buffers[i]) - valid_history_length
            action_history = ""
            action_history_with_think_arr = []
            for j, record in enumerate(recent_history):
                step_number = start_index + j + 1
                action = record["action"]
                env_obs = record["text_obs"]
                if "thinking_trace" in record.keys():
                    thinking_trace = record["thinking_trace"]
                else:
                    thinking_trace = "None or invalid thinking trace."
                action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                action_history_with_think_arr.append(f"\n[Observation {step_number}: '{env_obs}', Thoughts {step_number}: '{thinking_trace}' Action {step_number}: '{action}']")
            action_history_with_think = "".join(action_history_with_think_arr)

            # Check the token length
            attempts = 0
            if self.tok is not None:
                token_length = len(self.tok.encode(action_history_with_think))
                # If the token length is too long, we need to truncate the history (1000 is just for an extra safety margin)
                while token_length >= (self.config['actor_rollout_ref']['actor']['ppo_max_token_len_per_gpu'] - 1000):
                    action_history_with_think_arr.pop(0)
                    action_history_with_think = "".join(action_history_with_think_arr)
                    token_length = len(self.tok.encode(action_history_with_think))
                    attempts += 1
                    # If attempts go above 4/5 of the max steps, just default to the regular action history.
                    if attempts > (self.config['env']['max_steps'] * 0.8):
                        print(f"Warning: Too many attempts to truncate the action history for {self.env_name}. Defaulting to regular action history.")
                        action_history_with_think = action_history
                        break

            if self.config['env']['prompt_template'] == "basecase":
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_BASECASE
            elif self.config['env']['prompt_template'] == 'sctq':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ
            elif self.config['env']['prompt_template'] == 'ttp':
                if self.istrain:
                    if self.ttp_switch:
                        LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_TRAIN
                    else:
                        LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_TEST
                    if self.print_counter:
                        print(f"Using training template for step {len(self.buffers[i]) + 1}")
                        self.print_counter = False
                else:
                    LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_TEST
            elif self.config['env']['prompt_template'] == 'sctq_longer':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER
            elif self.config['env']['prompt_template'] == 'sctq_longer_warning':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER_WARNING
            elif self.config['env']['prompt_template'] == 'sctq_longer_no_warning':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER_NO_WARNING
            elif self.config['env']['prompt_template'] == 'no_warning':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_NO_WARNING
            elif self.config['env']['prompt_template'] == 'think_then_scq_longer':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_THINK_THEN_SCQ_LONGER
            elif self.config['env']['prompt_template'] == 'instructions_first_sctq':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST
            elif self.config['env']['prompt_template'] == 'instructions_first_sctq_more_info':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST_MORE_INFO
            elif self.config['env']['prompt_template'] == 'instructions_first_sctq_more_info_with_think':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST_MORE_INFO_WITH_THINK
            elif self.config['env']['prompt_template'] == 'instructions_first_more_info_with_think':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_INSTRUCTIONS_FIRST_MORE_INFO_WITH_THINK
            elif self.config['env']['prompt_template'] == 'instructions_first_more_info_no_scqt':
                LIFEGATE_TEMPLATE = LIFEGATE_TEMPLATE_INSTRUCTIONS_FIRST_NO_SCQRT
            else:
                raise ValueError(f"Unknown prompt template: {self.config.env.prompt_template}")

            obs = LIFEGATE_TEMPLATE.format(
                task_description=self.tasks[i],
                step_count=len(self.buffers[i]),
                history_length=valid_history_length,
                action_history=action_history.strip(),
                action_history_with_think=action_history_with_think.strip(),
                current_step=len(self.buffers[i]) + 1,
                current_observation=text_obs[i],
                admissible_actions=""
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, thinking_trace):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'thinking_trace': thinking_trace[i]})

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, env_name):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.buffers = None
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if init or history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    if self.is_multi_modal:
                        action_history += f"\n[Action {step_number}: '{record['action']}']"
                    else:
                        action_history += f"\n[Text Observation {step_number}: \n{record['text_obs']}\nAction {step_number}: '{record['action']}']"

                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': self.ACTION_LOOKUP[actions[i]]})


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False, history_length: int = 20) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nObservation {step_number}: \n{env_obs}\nCode {step_number}: \n{action}\n"
                
                if len(action_history) > 50000:
                    print(f"Warning len(action_history)={len(action_history)} is too long")
                    action_history = action_history[-50000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # Modified by Chris 6/26
    # Trying to strip this down as much as possible to just get started with ppo. Sort of based on alfworldTWEnv.
    if "tales_" in config.env.env_name.lower():
        from agent_system.environments.env_package.general_tag import build_general_envs, general_projection
        # Get the specific environment:
        target_env = config.env.env_name.split("tales_")[-1]
        yaml_filepath = os.path.join(os.path.dirname(__file__), 'env_package/general_tag/configs', f'config.yaml')
        _envs = build_general_envs(yaml_filepath, seed=config.env.seed, env_num=config.data.train_batch_size, 
                                       group_n=group_n, main_config=config, is_train=True)
        print("Training environments built successfully.")
        _val_envs = build_general_envs(yaml_filepath, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, 
                                           group_n=1, main_config=config, is_train=False)
    
        projection_f = partial(general_projection)
        envs = GeneralEnvironmentManager(_envs, projection_f, target_env, config=config)
        val_envs = GeneralEnvironmentManager(_val_envs, projection_f, target_env, istrain=False, config=config)
            
        return envs, val_envs
    elif 'lifegate' in config.env.env_name.lower():
        from agent_system.environments.env_package.lifegate import build_lifegate_envs, lifegate_projection

        yaml_filepath = os.path.join(os.path.dirname(__file__), 'env_package/lifegate/configs/config.yaml')
        _envs = build_lifegate_envs(lifegate_config_path=yaml_filepath, seed=config.env.seed, env_num=config.data.train_batch_size, 
                                    group_n=group_n, main_config = config, is_train=True)
        _val_envs = build_lifegate_envs(lifegate_config_path=yaml_filepath, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, 
                                        group_n=1, main_config = config, is_train=False)
        
        projection_f = partial(lifegate_projection)
        envs = LifeGateEnvironmentManager(_envs, projection_f, config.env.env_name, config=config)
        val_envs = LifeGateEnvironmentManager(_val_envs, projection_f, config.env.env_name, istrain=False, config=config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


if __name__ == "__main__":
    env_name = "lifegate"
    if env_name == "tales_alfworld":
        # Test LifegateEnvironmentManager
        from agent_system.environments.env_package.general_tag import build_general_envs, general_projection
        import time
        yaml_filepath = os.path.join(os.path.dirname(__file__), 'env_package/general_tag/configs/config.yaml')
        env_num = 4
        group_n = 5
        time1 = time.time()
        envs = build_general_envs(yaml_filepath, seed=1, env_num=env_num, group_n=group_n)
        env_manager = LifeGateEnvironmentManager(envs, lifegate_projection, 'lifegate/LifeGateEnv')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        random_actions_set = ['go north', 'go south', 'go east', 'go west']
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                print("step: ", i)
                random_actions = [np.random.choice(random_actions_set) for _ in range(len(infos))]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
            print("completed")
    elif env_name == "lifegate":
        # Test LifegateEnvironmentManager
        from agent_system.environments.env_package.lifegate import build_lifegate_envs, lifegate_projection
        import time
        yaml_filepath = os.path.join(os.path.dirname(__file__), 'env_package/lifegate/configs/config.yaml')
        env_num = 4
        group_n = 5
        time1 = time.time()
        envs = build_lifegate_envs(yaml_filepath, seed=1, env_num=env_num, group_n=group_n)
        env_manager = LifeGateEnvironmentManager(envs, lifegate_projection, 'lifegate/LifeGateEnv')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        random_actions_set = ['go north', 'go south', 'go east', 'go west']
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                print("step: ", i)
                random_actions = [np.random.choice(random_actions_set) for _ in range(len(infos))]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
            print("completed")
    elif env_name == "gym_cards":
        # Test GymCardEnvironmentManager
        env_num = 2
        group_n = 5
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        envs = build_gymcards_envs('gym_cards/EZPoints-v0', 0, env_num, group_n)
        projection_f = partial(gym_projection, env_name='gym_cards/EZPoints-v0')
        env_manager = GymCardEnvironmentManager(envs, projection_f, 'gym_cards/EZPoints-v0')
        obs, infos = env_manager.reset()
        for i in range(100):
            random_actions = [f'"action": {np.random.randint(0, 10)}' for i in range(len(infos))]
            obs, rewards, dones, infos = env_manager.step(random_actions)
            env_manager.save_image(obs['image'], i)
        print("completed")
    elif env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        import time
        alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        env_num = 8
        group_n = 5
        time1 = time.time()
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n)
        # val_envs = build_alfworld_envs(alf_config_path, 1000, 4)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = [np.random.choice(env_manager.envs.get_admissible_commands[i]) for i in range(len(env_manager.envs.get_admissible_commands))]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
            print("completed")

    elif env_name == "sokoban":
        # Test SokobanEnvironmentManager
        from agent_system.environments.env_package.sokoban import sokoban_projection
        from agent_system.environments.env_package.sokoban import build_sokoban_envs
        env_num = 2
        group_n = 5
        env_kwargs = {
            'dim_room': (6, 6),
            'num_boxes': 1,
            'max_steps': 100,
            'search_depth': 30
        }
        action_pools = {
            1: "<action>up</action>",
            2: "<action>down</action>",
            3: "<action>left</action>",
            4: "<action>right</action>",
        }
        # ['tiny_rgb_array', 'list', 'state', 'rgb_array']
        envs = build_sokoban_envs(0, env_num, group_n, mode='rgb_array', is_train=True, env_kwargs=env_kwargs)
        projection_f = partial(sokoban_projection)
        env_manager = SokobanEnvironmentManager(envs, projection_f, 'sokoban')
        obs, infos = env_manager.reset()
        for i in range(100):
            random_actions = [action_pools[np.random.randint(1, 5)] for i in range(len(infos))]
            obs, rewards, dones, infos = env_manager.step(random_actions)
            if obs['image'] is not None:
                env_manager.save_image(obs['image'][0], i)
            if np.array(dones).any():
                print("Episode completed")
    elif env_name == "webshop":
        # Test WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import webshop_projection
        from agent_system.environments.env_package.webshop import build_webshop_envs
        from agent_system.environments.env_package.webshop.webshop.web_agent_site.models import RandomPolicy
        import time
        env_num = 2
        group_n = 5
        time1 = time.time()
        file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
        attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': False,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        envs = build_webshop_envs(seed=1, env_num=env_num, group_n=group_n, env_kwargs=env_kwargs, is_train=True)
        # val_envs = build_webshop_envs(1000, 4)
        env_manager = WebshopEnvironmentManager(envs, webshop_projection, 'webshop')
        policy = RandomPolicy()
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = ['<action>'+policy.forward(None, info['available_actions'])+'</action>' for info in infos]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
        print("completed")

    elif env_name == "appworld":
        # Test AppWorldEnvironmentManager
        from agent_system.environments.env_package.appworld import appworld_projection
        from agent_system.environments.env_package.appworld import build_appworld_envs
        import time
        env_num = 2
        group_n = 5
        time1 = time.time()
        envs = build_appworld_envs(dataset_name='test_normal', max_interactions=50, seed=1, env_num=env_num, group_n=group_n)
        # val_envs = build_alfworld_envs(alf_config_path, 1000, 4)
        env_manager = AppWorldEnvironmentManager(envs, appworld_projection, 'appworld')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = ["print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))" for i in range(len(obs['text']))]
                # print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
        print("completed")