import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random

class linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='Snake_model.pth'):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        file_path = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr + random.uniform(-0.0005, 0.01)
        # self.lr = lr +
        self.gamma = gamma + random.uniform(-0.2, 0.09999999999)
        # self.gamma = gamma +
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        print(f"Trainer initialized with lr: {self.lr}, gamma: {self.gamma}")
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # Convert to batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Q value for current state
        pred = self.model(state)

        # Q value for next state
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # Train the model
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()