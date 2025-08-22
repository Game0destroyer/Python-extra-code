import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import linear_QNet, QTrainer
from helper import plot  # Assuming you have a plotting function in helpers.py

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        import os
        self.n_games = 0
        self.epsilon = 0  # control exploration
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = linear_QNet(input_size=11, hidden_size=256, output_size=3)  # Adjust input and output sizes as needed
        self.best_record = 0
        if os.path.exists('model/best_score.txt'):
            with open('model/best_score.txt', 'r') as f:
                num = str(f.read().strip())
        else:
            num = '0'
        if os.path.exists('model\Snake_model'+num+'.pth'):
            # If the file exists, read the number
            self.model.load_state_dict(torch.load('model\Snake_model'+num+'.pth'))
            print('Loaded existing model weights.')
            # Try to load best record from a file, or set to 0 if not found
            try:
                with open('model/best_score.txt', 'r') as f:
                    self.best_record = int(f.read().strip())
            except Exception:
                self.best_record = 0
        else:
            print('No existing model found. Starting fresh.')
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Initialize trainer with model and parameters
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food is left
            game.food.x > game.head.x,  # Food is right
            game.food.y < game.head.y,  # Food is up
            game.food.y > game.head.y   # Food is down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if max length exceeded

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)  #TODO: Implement training step in trainer

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)  #TODO: Implement training step in trainer

    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # more games, less exploration
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    record = agent.best_record
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                with open('model/best_score.txt', 'w') as f:
                    f.write(str(record))
                with open('model/best_score.txt', 'r') as f:
                    num = str(f.read().strip()) # If the file exists, read the number
                agent.model.save('Snake_model'+num+'.pth')  # Save the model if a new record is achieved
                # Save the new best record to a file
               
            
            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()