import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from torchviz import make_dot

# originally from Omkar Prabu's excellent intro to ggml: https://omkar.xyz/intro-ggml/

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 10000
LR = 0.01
MODEL_FILE = "model.pth"
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = {
    "xor": [[0], [1], [1], [0]],
    "or": [[0], [1], [1], [1]],
    "and": [[0], [0], [0], [1]],
}

gate_type = "xor"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


inputs = torch.tensor(INPUTS, device=device, dtype=torch.float32)
labels = torch.tensor(OUTPUTS[gate_type], device=device, dtype=torch.float32)

model = Model()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

print(f"Training for {NUM_EPOCHS} epochs..")
for epoch in tqdm(range(NUM_EPOCHS)):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), MODEL_FILE)

# make_dot(model(inputs)).render("model", format="png", cleanup=True)

with torch.no_grad():
    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    print(inputs)
    print(labels)
    print(model(inputs).cpu().numpy())
    outputs = model(torch.tensor([0, 0], device=device, dtype=torch.float32))
    print(outputs.cpu().numpy())
