# Double-Q Learning AI Agent for Mario

This project aims to develop a Double-Q Learning AI agent to play Mario using the Gymnasium library and the Atari version of MarioBros.

## Project Structure

```
project-dir
|-- .
|-- ./paper
|-- ./requirements.txt
|-- ./README.md
|-- ./result
|-- ./src
|   |-- ./src/env
|   |   |-- ./src/env/wrappers.py
|   |   |-- ./src/env/__init__.py
|   |-- ./src/utils
|   |   |-- ./src/utils/__init__.py
|   |   |-- ./src/utils/logger.py
|   |-- ./src/agent
|   |   |-- ./src/agent/mario.py
|   |   |-- ./src/agent/__init__.py
|   |   |-- ./src/agent/net.py
|-- ./notebook
|   |-- ./notebook/gym_exploration.ipynb
```

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

To explore the Gymnasium environment, open and run the `gym_exploration.ipynb` notebook:

```bash
jupyter notebook notebook/gym_exploration.ipynb
```

### Training the Agent

The main training script is located in `src/agent/`. To train the agent, run the following command:

```bash

```

## Wrappers

The environment wrappers are defined in `src/env/wrappers.py`. These wrappers are used to preprocess the environment observations and actions.

## Logging

Logging utilities are provided in `src/utils/logger.py` to log the training progress and results.

## Results

The results of the training are stored in the `result` directory.

## Paper

The project report and analysis are stored in the `paper` directory.
