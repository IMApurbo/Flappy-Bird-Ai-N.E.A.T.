# Flappy Bird AI using NEAT (Neuroevolution of Augmenting Topologies)

This project implements a Flappy Bird game where an AI agent is trained using NEAT (Neuroevolution of Augmenting Topologies) to learn how to play the game. The AI uses a neural network to make decisions such as when to jump to avoid pipes. The model is evolved over multiple generations to improve performance.

## Requirements

- Python 3.x
- Pygame
- NEAT-Python (for NEAT algorithm)
- Pickle (for saving/loading models)

## Installation

To run this project, you need to install the required dependencies. You can do so by running:

```bash
pip install pygame neat-python
```

Make sure you also have Pygame installed for rendering the game graphics.

## Project Structure

- **flappy_bird.py**: The main script that runs the game and trains the AI.
- **background.png**: Image used as the background for the game.
- **bird.png**: Image used for the bird character.
- **pipe_top.png**: Image used for the top part of the pipes.
- **neat-config.txt**: Configuration file used by NEAT for defining the neural network structure and training parameters.

## How to Train the AI

### 1. Start the Training

Run the following command to start training the AI to play Flappy Bird using NEAT:

```bash
python flappy-bird.py --train
```

This will start the training process. The AI will attempt to navigate the bird through pipes by learning over several generations. The progress of the training will be displayed in the terminal.

### 2. Interrupt and Save the Model

You can stop the training at any time by pressing `Ctrl + C`. The model weights will be saved in a pickle file. This file can then be used to test the AI or continue training later.

## How to Test a Saved Model

To test a saved model (which you may have trained previously), use the following command:

```bash
python flappy-bird.py --model <path_to_model.pkl>
```

This will load the saved model and run the Flappy Bird game where the AI will try to play based on the model’s learned behavior.

## Game Description

- The AI learns to control the bird's movement to avoid pipes.
- The bird has to avoid colliding with the pipes, which randomly appear at varying heights and move horizontally across the screen.
- The game uses the NEAT algorithm to evolve the neural network, allowing the bird to make decisions like when to jump.

## Files to Be Used

1. **flappy-bird.py**: The script used to run the game and train the AI.
2. **background.png**: The background image for the Flappy Bird game.
3. **bird.png**: The image of the bird that the AI controls.
4. **pipe_top.png**: The image for the top part of the pipe.
5. **neat-config.txt**: The NEAT configuration file that defines the structure of the neural network.

## Conclusion

This project demonstrates how the NEAT algorithm can be used to evolve neural networks capable of playing games, using the Flappy Bird game as a test case. You can modify the NEAT configuration, game logic, or add more features for further experimentation.
