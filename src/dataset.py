import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

class ISICDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, groundtruth_csv, tokenizer, transform=None, limit_samples=None):
        self.image_dir = image_dir
        self.metadata_df = pd.read_csv(metadata_csv)
        self.labels_df = pd.read_csv(groundtruth_csv)
        self.tokenizer = tokenizer
        self.transform = transform

        # Merge using 'image_name' as the key
        self.df = pd.merge(self.labels_df, self.metadata_df, on='image_name', how='inner')

        # Limit the dataset to specified number of samples
        if limit_samples is not None and limit_samples < len(self.df):
            self.df = self.df.head(limit_samples)
            print(f"Dataset limited to {limit_samples} samples")

        # Print available columns after merge for debugging
        print("Final merged columns:", self.df.columns.tolist())
        print(f"Final dataset size: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, row['image_name'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Extract clinical metadata from merged columns (with _x suffix)
        site = row.get('anatom_site_general_challenge_x', 'unknown site')
        age = row.get('age_approx_x', 'unknown age')
        sex = row.get('sex_x', 'unknown sex')

        # Construct clinical text
        clinical_text = f"A lesion from the {site} of a {age}-year-old {sex}."
        encoding = self.tokenizer(
            clinical_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        # Get label from target_x column
        label = torch.tensor(row['target_x'], dtype=torch.long)

        return image, encoding, label
