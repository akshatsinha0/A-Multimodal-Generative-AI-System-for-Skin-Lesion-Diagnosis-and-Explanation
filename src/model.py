import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
from transformers import BertModel

# --------- Image Encoder ---------
class ImageEncoder(nn.Module):
    def __init__(self, freeze=True):
        super(ImageEncoder, self).__init__()
        # Use updated weights API
        weights = EfficientNet_B3_Weights.DEFAULT
        efficientnet = models.efficientnet_b3(weights=weights)

        self.backbone = efficientnet.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = efficientnet.classifier[1].in_features  # 1536

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        return x  # shape: (batch_size, 1536)

# --------- Text Encoder ---------
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', freeze=True):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.feature_dim = 768  # CLS token size

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding  # shape: (batch_size, 768)

# --------- Fusion Module ---------
class FusionModule(nn.Module):
    def __init__(self, image_dim, text_dim, fusion_dim=512):
        super(FusionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, image_feat, text_feat):
        combined = torch.cat((image_feat, text_feat), dim=1)
        return self.fusion(combined)  # shape: (batch_size, fusion_dim)

# --------- Multimodal Classifier ---------
class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.fusion_module = FusionModule(
            image_dim=self.image_encoder.feature_dim,
            text_dim=self.text_encoder.feature_dim
        )
        self.classifier = nn.Linear(512, 2)  # Binary classification (benign/malignant)

    def forward(self, image, text_input):
        image_feat = self.image_encoder(image)
        text_feat = self.text_encoder(text_input['input_ids'], text_input['attention_mask'])
        fused = self.fusion_module(image_feat, text_feat)
        out = self.classifier(fused)
        return out
