import argparse
import torch
from torch.utils.data import DataLoader
from models.diffsr_modules import RDN  # Import your RDN model
from tasks.srdiff_brats import BraTSDataset  # Import your BraTS dataset class
from utils.sr_utils import SSIM, L1CharbonnierLoss  # Import utility functions

class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RDN(in_channels=args.in_channels, out_channels=args.out_channels, num_blocks=args.num_blocks, growth_rate=args.growth_rate, num_layers=args.num_layers)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = L1CharbonnierLoss()

    def train(self, dataset, epochs, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for data in dataloader:
                inputs = data.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)
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
    parser = argparse.ArgumentParser(description="Train and evaluate RDN model on BraTS dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels")
    parser.add_argument("--num_blocks", type=int, default=5, help="Number of RDN blocks")
    parser.add_argument("--growth_rate", type=int, default=32, help="Growth rate for RDN")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in each RDN block")
    args = parser.parse_args()

    dataset = BraTSDataset(data_dir='path/to/your/data')
    trainer = Trainer(args)
    trainer.train(dataset, args.epochs, args.batch_size)
    trainer.evaluate(dataset)
