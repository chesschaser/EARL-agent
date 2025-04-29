"""
EARL (Evolutionary Adaptive Reinforcement Learner)

A lightweight agent that evolves probabilistic action weights based on feedback from a fitness function.
"""

import random
import pickle

def save_agent(agent, file):
    with open(file, "wb") as f:
        pickle.dump(agent, f)

def load_agent(file):
    with open(file, "rb") as f:
        return pickle.load(f)
    
class EARL_Agent:
    """
    EARL_Agent adapts its action selection strategy over time using evolutionary reinforcement learning.

    Args:
        action_space (list): A list of callable actions.
        fitness_func (callable): Returns a numerical fitness score.
        weight_variance_threshold (float): Threshold for detecting insufficient exploration.
        mutation_strength (float): How much mutation changes the action weights.
        min_mutation_step (int): Fastest mutation interval.
        max_mutation_step (int): Slowest mutation interval.
        history_bias_strength (float): Influences how strongly past success impacts learning.
        history_decay_factor (float): How quickly historical bias fades over time.
        alpha (float): Learning rate scaling factor.
    """
        
    def __init__(self, action_space, fitness_func, weight_variance_threshold=0.5, mutation_strength=0.1, min_mutation_step=3, max_mutation_step=10, history_bias_strength=0.1, history_decay_factor=0.9, alpha=1):
        self.action_space = action_space
        self.weight_map = self._generate_weight_map()

        self.alpha = alpha
        
        self.fitness_func = fitness_func        
        self.last_fitness = self.fitness_func()
        
        self.mutation_step = min_mutation_step
        self.weight_variance_threshold = weight_variance_threshold
        self.min_mutation_step = min_mutation_step
        self.max_mutation_step = max_mutation_step
        self.mutation_strength = mutation_strength
        
        self.history = [0 for action in self.action_space]
        self.history_bias_strength = history_bias_strength
        self.history_decay_factor = history_decay_factor
        
        self.ticks = 0

    def _normalize(self, l):
        """
        Normalizes a list of numbers so that their sum is 1.
        
        Returns uniform distribution if the total sum is zero.
        """
        
        total = sum(l)
        
        if total:
            return [i / total for i in l]
        else:
            n = len(l)
            return [1 / n for _ in range(n)]
        
    def _variance(self, l):
        """
        Computes the population variance of a list of numbers.

        Returns 0 if the list is empty.
        """
        
        if not l:
            return 0
        mean = sum(l) / len(l)
        return sum((x - mean) ** 2 for x in l) / len(l)
    
    def _generate_weight_map(self):
        """
        Creates an initial random weight map over the action space,
        normalized to form a probability distribution.
        """
        
        weight_map = [random.random() for action in self.action_space]
        return self._normalize(weight_map)

    def tick(self):
        """
        Performs a single time step of the agent's learning cycle.

        Includes weight map mutation, action selection, fitness evaluation,
        weight adjustment, and history tracking.
        """
        
        self._adjust_mutation_step()

        if not self.ticks % self.mutation_step:
            self._mutate_weight_map()

        indices = []
        for i, weight in enumerate(self.weight_map):
            if random.random() < weight:
                indices.append(i)
                self.action_space[i]()

        fitness = self.fitness_func()

        delta_fitness = fitness - self.last_fitness
        self._adjust_weight_map(delta_fitness, indices)

        self.last_fitness = fitness
                
        if delta_fitness > 0:
            for index in indices:
                self.history[index] += 1
        else:
            for index in indices:
                self.history[index] = 0

        self.ticks += 1

    def _adjust_mutation_step(self):
        """
        Dynamically adjusts the mutation interval based on weight diversity.

        Decreases mutation rate if variance is high, increases it if low.
        """
        
        if self._variance(self.weight_map) < self.weight_variance_threshold:
            self.mutation_step = max(self.min_mutation_step, self.mutation_step - 1)
        else:
            self.mutation_step = min(self.max_mutation_step, self.mutation_step + 1)

    def _mutate_weight_map(self):
        """
        Applies random mutations to all action weights.

        Each weight is adjusted by a small random amount within ±mutation_strength,
        and then clipped to stay within the [0, 1] range.
        """
        
        for i in range(len(self.weight_map)):
            self.weight_map[i] = min(1, max(0, self.weight_map[i] + random.uniform(-self.mutation_strength, self.mutation_strength)))
    
    def _adjust_weight_map(self, delta_fitness, indices):
        """
        Updates weights for actions taken during this tick based on fitness change.

        Applies historical biasing and learning rate scaling (alpha).
        Re-normalizes weights to maintain a valid probability distribution.
        """
        
        if not indices:
            return

        weight_modifier = delta_fitness / len(indices)

        for index in indices:
            self.history[index] *= self.history_decay_factor
            history_bias = 1 + (self.history[index] * self.history_bias_strength)
            self.weight_map[index] = min(1, max(0, self.weight_map[index] + (weight_modifier * self.alpha * history_bias)))
            
        self.weight_map = self._normalize(self.weight_map)
