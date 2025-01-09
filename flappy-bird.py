import pygame
import random
import sys
import neat
import pickle
import argparse

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Game settings
FPS = 60
GRAVITY = 0.3
JUMP_STRENGTH = -8
PIPE_SPEED = 3
PIPE_GAP = 150

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Clock
clock = pygame.time.Clock()

# Load assets
font = pygame.font.Font(None, 36)

# Load images
background_image = pygame.transform.scale(pygame.image.load("background.png"), (WIDTH, HEIGHT))
bird_image = pygame.image.load("bird.png")
pipe_image = pygame.image.load("pipe_top.png")

# Bird class
class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 30
        self.velocity = 0
        self.image = pygame.transform.scale(bird_image, (self.width, self.height))

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def jump(self):
        self.velocity = JUMP_STRENGTH

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(50, HEIGHT - PIPE_GAP - 50)
        self.bottom_height = HEIGHT - self.top_height - PIPE_GAP
        self.pipe_width = 50

        # Scale the top and bottom parts of the pipe dynamically
        self.pipe_top = pygame.transform.scale(pipe_image, (self.pipe_width, self.top_height))
        self.pipe_bottom = pygame.transform.flip(
            pygame.transform.scale(pipe_image, (self.pipe_width, self.bottom_height)),
            False, True
        )

    def draw(self):
        # Top pipe
        screen.blit(self.pipe_top, (self.x, 0))
        # Bottom pipe
        screen.blit(self.pipe_bottom, (self.x, HEIGHT - self.bottom_height))

    def update(self):
        self.x -= PIPE_SPEED

# Evaluate genome
def eval_genomes(genomes, config):
    nets = []
    birds = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(100, HEIGHT // 2))
        genome.fitness = 0
        ge.append(genome)

    pipes = [Pipe(WIDTH + 200)]
    score = 0
    running = True

    while running and len(birds) > 0:
        screen.blit(background_image, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                # Save the best model when 'q' is pressed
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(nets[0], f)
                print("Model saved to best_model.pkl")
                running = False

        # Update pipes
        for pipe in pipes:
            pipe.update()
            if pipe.x + pipe.pipe_width < 0:
                pipes.remove(pipe)
                pipes.append(Pipe(WIDTH + 200))
                score += 1
                for genome in ge:
                    genome.fitness += 5

        # Check collisions and bird updates
        for i, bird in enumerate(birds):
            output = nets[i].activate((bird.y, abs(bird.y - pipes[0].top_height), abs(bird.y - (HEIGHT - pipes[0].bottom_height))))

            if output[0] > 0.5:
                bird.jump()

            bird.update()

            if bird.y + bird.height > HEIGHT or bird.y < 0:
                ge[i].fitness -= 1
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

            for pipe in pipes:
                if (bird.x + bird.width > pipe.x and bird.x < pipe.x + pipe.pipe_width and
                    (bird.y < pipe.top_height or bird.y + bird.height > HEIGHT - pipe.bottom_height)):
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)

        # Draw bird and pipes
        for bird in birds:
            bird.draw()
        for pipe in pipes:
            pipe.draw()

        # Draw score
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

# Test a saved model
def test_model(model_path):
    # Load the saved model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    bird = Bird(100, HEIGHT // 2)
    pipes = [Pipe(WIDTH + 200)]
    running = True
    score = 0

    while running:
        screen.blit(background_image, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update pipes
        for pipe in pipes:
            pipe.update()
            if pipe.x + pipe.pipe_width < 0:
                pipes.remove(pipe)
                pipes.append(Pipe(WIDTH + 200))
                score += 1

        # Bird decision-making
        output = model.activate((bird.y, abs(bird.y - pipes[0].top_height), abs(bird.y - (HEIGHT - pipes[0].bottom_height))))
        if output[0] > 0.5:
            bird.jump()

        bird.update()

        # Check for collisions
        if bird.y + bird.height > HEIGHT or bird.y < 0:
            print(f"Game Over! Final Score: {score}")
            running = False

        for pipe in pipes:
            if (bird.x + bird.width > pipe.x and bird.x < pipe.x + pipe.pipe_width and
                (bird.y < pipe.top_height or bird.y + bird.height > HEIGHT - pipe.bottom_height)):
                print(f"Game Over! Final Score: {score}")
                running = False

        # Draw bird and pipes
        bird.draw()
        for pipe in pipes:
            pipe.draw()

        # Draw score
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--model", type=str, help="Path to the saved model for testing")
    args = parser.parse_args()

    if args.train:
        config_path = "neat-config.txt"
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_path)

        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        winner = population.run(eval_genomes, 50)

    elif args.model:
        test_model(args.model)
    else:
        print("Please provide either --train to train the model or --model=<path> to test a saved model.")
