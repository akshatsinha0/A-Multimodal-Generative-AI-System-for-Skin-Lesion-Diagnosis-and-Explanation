import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                             

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm

from dataset import ISICDataset
from model   import MultimodalClassifier


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ISIC multimodal training")
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit training to N images (debug). Omit for full dataset."
    )
    p.add_argument(
        "--epochs", type=int, default=5,
        help="Number of epochs."
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size."
    )
    return p.parse_args()


# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Paths (adjust only if your folder names differ)
    IMAGE_DIR       = "/content/data/images/train"
    METADATA_CSV    = "/content/data/ISIC_2020_Training_Metadata_v2.csv"
    GROUNDTRUTH_CSV = "/content/data/ISIC_2020_Training_GroundTruth.csv"

    CKPT_DIR = Path("/content/drive/MyDrive/model_ckpts")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Tokenizer & image transforms
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    # ------------------------------------------------------------------
    # Dataset & loader
    dataset = ISICDataset(
        IMAGE_DIR,
        METADATA_CSV,
        GROUNDTRUTH_CSV,
        tokenizer,
        transform,
        limit_samples=args.max_samples  # None == full set
    )
    print("Dataset size:", len(dataset))
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)

    # ------------------------------------------------------------------
    # Model, loss, optimiser
    model = MultimodalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, clin_enc, labels in pbar:
            images  = images.to(device)
            labels  = labels.to(device)
            clin_enc = {k: v.to(device) for k, v in clin_enc.items()}

            optimizer.zero_grad()
            outputs = model(images, clin_enc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch}/{args.epochs} finished – avg loss: {avg_loss:.4f}")

        # save after each epoch (optional but safer)
        epoch_ckpt = CKPT_DIR / f"epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_ckpt)

    # ------------------------------------------------------------------
    # Final checkpoint
    final_ckpt = CKPT_DIR / "model_weights_latest.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"✅ Training complete – final weights saved to {final_ckpt}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
