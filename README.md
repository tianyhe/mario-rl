# MarioRL

A deep reinforcement learning agent that masters Super Mario Bros using advanced techniques like Dueling DDQN and prioritized experience replay.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Exploring the Environment](#exploring-the-environment)
  - [Training the Agent](#training-the-agent)
- [Components](#components)
  - [Environment Wrappers](#environment-wrappers)
  - [Agent](#agent)
  - [Neural Network Models](#neural-network-models)
  - [Replay Buffer](#replay-buffer)
  - [Metric Logging](#metric-logging)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

MarioRL is a game AI project that aims to develop a deep reinforcement learning agent capable of mastering the classic game Super Mario Bros. The project utilizes the Gymnasium library and the Atari version of Super Mario Bros, implementing state-of-the-art techniques such as Dueling Double Deep Q-Networks (Dueling DDQN) and prioritized experience replay.

## Features

- Support for multiple neural network models (DQN, Dueling DQN, DDQN, Dueling DDQN)
- Default: Dueling DDQN architecture for improved learning efficiency
- Prioritized experience replay for focusing on important experiences
- Custom environment wrappers for preprocessing game states
- Comprehensive metric logging and visualization
- Jupyter notebooks for easy experimentation and training

## Project Structure

```
project-dir
|-- paper/
|-- requirements.txt
|-- README.md
|-- result/
|-- src/
|   |-- env/
|   |   |-- wrappers.py
|   |   |-- __init__.py
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- metric_logger.py
|   |-- agent/
|   |   |-- mario.py
|   |   |-- __init__.py
|   |   |-- net.py
|   |   |-- replay_buffer.py
|-- notebook/
|   |-- results/
|   |-- gym_exploration.ipynb
|   |-- train.ipynb
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MarioRL.git
   cd MarioRL
   ```

2. Set up the environment:
   - Using Conda:
     ```
     conda env create -f environment.yml
     conda activate mario-rl
     ```
   - Using pip:
     ```
     pip install -r requirements.txt
     ```

Note: The Conda environment is recommended as it ensures all dependencies are correctly resolved.

## Usage

### Exploring the Environment

To explore the Gymnasium environment for Super Mario Bros:

```bash
jupyter notebook notebook/gym_exploration.ipynb
```

### Training the Agent

To train a new agent or use a pre-trained agent:

```bash
jupyter notebook notebook/train.ipynb
```

## Components

### Environment Wrappers

Located in `src/env/wrappers.py`, these custom wrappers preprocess the game environment:
- `SkipFrame`: Skips frames to reduce computation
- `GrayScaleObservation`: Converts observations to grayscale
- `ResizeObservation`: Resizes observations to a standard size

### Agent

The `Mario` class in `src/agent/mario.py` is the core of the learning agent, handling interactions with the environment and the learning process.

### Neural Network Models

Defined in `src/agent/net.py`, these include:
- `QNetworkCNN`: Traditional DQN network
- `QNetworkDuellingCNN`: Dueling DQN architecture

### Replay Buffer

Implemented in `src/agent/replay_buffer.py`, the `ReplayBuffer` class manages experience storage and sampling, supporting prioritized experience replay.

### Metric Logging

The `MetricLogger` class in `src/utils/metric_logger.py` handles logging and visualization of training metrics.

## Results

Training results and generated plots are stored in the `notebook/results` directory. Detailed analysis and project reports can be found in the `paper` directory.

## Contributing

Contributions to MarioRL are welcome! Please feel free to submit a Pull Request.

## License

MarioRL is licensed under the MIT License. See the `LICENSE` file for more information.

---

For more information, please refer to the individual component files and Jupyter notebooks in the project.