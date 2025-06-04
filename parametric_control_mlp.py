import torch
import torch.nn as nn


class control_mlp(nn.Module):
    def __init__(self, embedding_size):
        super(control_mlp, self).__init__()

        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()

        self.edit_strength_fc1 = nn.Linear(1, 128)
        self.edit_strength_fc2 = nn.Linear(128, 2)

    def forward(self, x, edit_strength):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        edit_strength = self.relu(self.edit_strength_fc1(edit_strength.unsqueeze(1)))
        edit_strength = self.edit_strength_fc2(edit_strength)

        edit_strength1, edit_strength2 = edit_strength[:, 0], edit_strength[:, 1]
        # print(edit_strength1.shape)
        # exit()

        output = (
            edit_strength1.unsqueeze(1) * x[:, :1024]
            + edit_strength2.unsqueeze(1) * x[:, 1024:]
        )

        return output
