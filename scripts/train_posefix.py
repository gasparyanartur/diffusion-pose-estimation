# train_posefix.py

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

# ----------------------------
# Utilities
# ----------------------------

def joints_to_heatmap(joints, bbox, H=64, W=64, sigma=2):
    """
    Convert keypoints to heatmaps using person bbox for scaling.
    joints: [num_joints,3]
    bbox: [x, y, w, h]
    """
    num_joints = joints.shape[0]
    heatmaps = np.zeros((num_joints, H, W), dtype=np.float32)
    bx, by, bw, bh = bbox
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))

    for j in range(num_joints):
        x, y, v = joints[j]
        if v == 0:  # invisible
            continue

        # normalize to bbox
        x = (x - bx) * W / bw
        y = (y - by) * H / bh

        if x < 0 or y < 0 or x >= W or y >= H:
            continue

        heatmaps[j] = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma**2))
    return heatmaps

# ----------------------------
# Dataset for GPU-optimized training
# ----------------------------

class PoseFixDataset(Dataset):
    def __init__(self, gt_hm_file, pred_hm_file):
        self.gt_hm = torch.load(gt_hm_file)
        self.pred_hm = torch.load(pred_hm_file)
        assert len(self.gt_hm) == len(self.pred_hm), "GT and pred lengths mismatch"

    def __len__(self):
        return len(self.gt_hm)

    def __getitem__(self, idx):
        return self.pred_hm[idx], self.gt_hm[idx]

# ----------------------------
# PoseFix Model
# ----------------------------

class PoseFix(nn.Module):
    def __init__(self, num_joints=14):
        super().__init__()
        self.conv1 = nn.Conv2d(num_joints, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.head = nn.Conv2d(128, num_joints, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.head(x)

# ----------------------------
# Precompute heatmaps if missing
# ----------------------------

def precompute_heatmaps(gt_file, pred_file, H=64, W=64):
    gt_hm_path = "gt_heatmaps.pt"
    pred_hm_path = "pred_heatmaps.pt"

    if os.path.exists(gt_hm_path) and os.path.exists(pred_hm_path):
        print("Using existing precomputed heatmaps.")
        return gt_hm_path, pred_hm_path

    print("Precomputing heatmaps (this may take a while)...")
    with open(gt_file) as f:
        gt_json = json.load(f)
    with open(pred_file) as f:
        pred_json = json.load(f)

    gt_annotations = gt_json["annotations"]

    # group predictions by image_id
    preds_by_image = {}
    for p in pred_json:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    all_gt_hm = []
    all_pred_hm = []

    for ann in gt_annotations:
        img_id = ann["image_id"]
        if img_id not in preds_by_image:
            continue

        # pick highest-score prediction
        pred = max(preds_by_image[img_id], key=lambda x: x["score"])

        bbox = ann["bbox"]
        gt_joints = np.array(ann["keypoints"]).reshape(-1,3)
        pred_joints = np.array(pred["keypoints"]).reshape(-1,3)

        gt_hm = joints_to_heatmap(gt_joints, bbox, H=H, W=W)
        pred_hm = joints_to_heatmap(pred_joints, bbox, H=H, W=W)

        all_gt_hm.append(torch.from_numpy(gt_hm).float())
        all_pred_hm.append(torch.from_numpy(pred_hm).float())

    torch.save(all_gt_hm, gt_hm_path)
    torch.save(all_pred_hm, pred_hm_path)
    print("Precomputed heatmaps saved!")
    return gt_hm_path, pred_hm_path

# ----------------------------
# Training
# ----------------------------

def main():
    data_root = Path("data/crowdpose")
    gt_file = data_root / "annotations/crowdpose_train.json"
    pred_file = Path("posefix_predictions/posefix_train_predictions.keypoints.json")

    num_joints = 14
    batch_size = 16
    num_epochs = 50
    lr = 1e-4
    heatmap_size = (64,64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # precompute/load heatmaps
    gt_hm_file, pred_hm_file = precompute_heatmaps(gt_file, pred_file, *heatmap_size)

    # dataset and loader
    dataset = PoseFixDataset(gt_hm_file, pred_hm_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # model, optimizer, loss
    model = PoseFix(num_joints=num_joints).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for pred_hm, gt_hm in loader:
            pred_hm = pred_hm.to(device)
            gt_hm = gt_hm.to(device)
            out = model(pred_hm)
            loss = criterion(out, gt_hm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pred_hm.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    # save model
    save_path = Path("posefix_crowdpose_hrnetdkr.pth")
    torch.save(model.state_dict(), save_path)
    print(f"PoseFix model saved to {save_path}")

if __name__ == "__main__":
    main()