import json
import torch
from torch.utils.data import Dataset

class PoseFixDataset(Dataset):

    def __init__(self, gt_file, pred_file):

        with open(gt_file) as f:
            self.gt = json.load(f)

        with open(pred_file) as f:
            self.pred = json.load(f)

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):

        pred_kpts = self.pred[idx]["keypoints"]
        gt_kpts = self.gt[idx]["keypoints"]

        pred_kpts = torch.tensor(pred_kpts).view(-1,3)
        gt_kpts = torch.tensor(gt_kpts).view(-1,3)

        return pred_kpts, gt_kpts