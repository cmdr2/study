import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

# from torchviz import make_dot

# originally from Omkar Prabu's excellent intro to ggml: https://omkar.xyz/intro-ggml/

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--gate-type", default="xor", choices=["xor", "and", "or"], help="The type of logic gate to train."
)
parser.add_argument(
    "--print-weights", action="store_true", default=False, help="Print the trained weights to the console."
)
parser.add_argument("-o", "--output-file", default="model.pth", help="The output file to write the trained weights to.")
parser.add_argument("-e", "--epochs", type=int, default=10000, help="The number of epochs to train.")
parser.add_argument("-lr", "--learning-rate", type=float, default=0.01, help="The learning rate.")
parser.add_argument("-d", "--device", default=None, help="The device to train on, e.g. 'cuda', 'cpu' etc.")

args = parser.parse_args()

if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = {
    "xor": [[0], [1], [1], [0]],
    "or": [[0], [1], [1], [1]],
    "and": [[0], [0], [0], [1]],
}


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
labels = torch.tensor(OUTPUTS[args.gate_type], device=device, dtype=torch.float32)

model = Model()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

print(f"Training for {args.epochs} epochs..")
for epoch in tqdm(range(args.epochs)):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.to("cpu").state_dict(), args.output_file)
print(f"Written the weights to {args.output_file}")

if args.print_weights:
    print("\n--- WEIGHTS ---\n")
    for key, val in model.state_dict().items():
        v = [f"{v:.8f}" for v in val.flatten()]
        print(key.replace(".", "_"), "= {", ", ".join(v), "}")
    print("\n--- /WEIGHTS ---\n")

# make_dot(model(inputs)).render("model", format="png", cleanup=True)

print("--- TEST INFERENCE ---")
with torch.no_grad():
    model = Model().to(device)
    model.load_state_dict(torch.load(args.output_file, weights_only=True))
    model.eval()

    print("inputs", inputs.tolist())
    print("expected labels", labels.tolist())
    print("actual labels", model(inputs).cpu().numpy().tolist())
print("--- /TEST INFERENCE ---")
