
# EARL: Evolutionary Adaptive Reinforcement Learner

EARL (Evolutionary Adaptive Reinforcement Learner) is a lightweight reinforcement learning agent that evolves its action probabilities based on feedback from a fitness function. It adapts its action-selection strategy over time to maximize performance.

## Features

- **Probabilistic Action Selection**: Actions are chosen based on dynamically adjusted probabilities.
- **Evolutionary Learning**: The agent evolves its strategy using a fitness-based approach to reinforce successful actions.
- **Mutation-Based Learning**: Action weights are mutated to explore new strategies.
- **History Biasing**: Past success influences future decisions with an adjustable decay factor.
- **Dynamic Mutation Step**: Mutation rate is dynamically adjusted based on weight variance, promoting exploration when necessary.

## Installation

To use EARL, clone this repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

Make sure to have Python installed on your system. No additional dependencies are required at the moment.

## Usage

Here’s an example of how to use the `EARL_Agent` class:

```python
from earl import EARL_Agent, save_agent, load_agent

class NumberGameEnvironment:
    def __init__(self):
        self.action_space = [self.increment, self.decrement]

        self.target = 100
        self.total = 0

        self.agent = EARL_Agent(self.action_space, self.fitness_func)

        self.max_fitness = 0

    def increment(self):
        self.total += 1

    def decrement(self):
        self.total -= 1

    def fitness_func(self):
        return -abs(self.target - self.total)

    def train(self, max_ticks=1000, episodes=100):
        episode = 0
        self.agent.ticks = 0

        while episode < episodes:
            self.total = 1
            while self.agent.ticks < max_ticks:
                if self.agent.last_fitness >= self.max_fitness:
                    break
                self.agent.tick()
            episode += 1

        save_agent(self.agent, "earl.pkl")

    def solve(self, max_ticks=1000):
        agent = load_agent("earl.pkl")
        
        agent.ticks = 0
        
        self.total = 1
        while agent.ticks < max_ticks:
            agent.tick()
            if agent.last_fitness >= self.max_fitness:
                print(f"Solved! ticks={agent.ticks}")
                return
        print("Timeout!")
            
env = NumberGameEnvironment()
env.solve()
```

## How It Works

- **Action Selection**: The agent selects actions probabilistically, with weights that evolve based on feedback.
- **Fitness Evaluation**: After each tick, the agent evaluates its fitness using a fitness function. The fitness score determines the agent's success.
- **Weight Adjustment**: The weights of the actions are adjusted using the results of the fitness evaluation.
- **Mutation**: Action weights mutate over time to explore new strategies. The mutation step is adjusted dynamically based on the weight variance.
- **History Bias**: Actions that were successful in the past are given more weight, encouraging the agent to select them more often. Historical bias decays over time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make changes, and create a pull request. Before contributing, please ensure that any changes are thoroughly tested.

- **Bug Reports**: If you encounter any bugs, please open an issue on GitHub.
- **Feature Requests**: If you have any ideas or feature requests, feel free to open an issue or submit a pull request.

## Acknowledgments

- Inspired by concepts in **evolutionary algorithms** and **reinforcement learning**.
- Thanks to the open-source community for the inspiration that helped in the development of this agent.
- Special thanks to anyone who has contributed to similar projects or provided feedback.

## TODO (Optional)

- Implement more advanced mutation options.
- Make it more robust to invalid arguments and the like.
- Expand the action space to support more complex actions (non-binary).

---
