import torch
from torch import optim
from torch.utils.data import DataLoader
from models.diffsr_modules import RDN  # Import your RDN model
from .srdiff_brats import BraTSDataset  # Import the BraTS dataset class
from utils.sr_utils import SSIM, L1CharbonnierLoss  # Import utility functions

class SRDiffTrainer:
    def __init__(self, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RDN(in_channels=3, out_channels=3, num_blocks=5, growth_rate=32, num_layers=4)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = L1CharbonnierLoss()  # Choose an appropriate loss function

    def train(self, dataset, epochs=10, batch_size=16):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for data in dataloader:
                inputs = data.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)  # Assuming a self-reconstruction setup
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()
        ssim_metric = SSIM()
        with torch.no_grad():
            for data in dataloader:
                inputs = data.to(self.device)
                outputs = self.model(inputs)
                ssim = ssim_metric(outputs, inputs)
                print(f"SSIM: {ssim.item()}")

if __name__ == "__main__":
    # Example usage
    dataset = BraTSDataset(data_dir='path/to/your/data')
    trainer = SRDiffTrainer()
    trainer.train(dataset)
    trainer.evaluate(dataset)
