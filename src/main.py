import os
from transformers import BertTokenizer
from torchvision import transforms
from dataset import ISICDataset

# Set file paths
image_dir = "../data/images/train"
metadata_csv = "../data/ISIC_2020_Training_Metadata_v2.csv"
groundtruth_csv = "../data/ISIC_2020_Training_GroundTruth.csv"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Initialize dataset
dataset = ISICDataset(
    image_dir=image_dir,
    metadata_csv=metadata_csv,
    groundtruth_csv=groundtruth_csv,
    tokenizer=tokenizer,
    transform=transform
)

print(f"Dataset size: {len(dataset)} samples")
