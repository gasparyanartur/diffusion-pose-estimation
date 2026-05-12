import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


class CrowdPoseRawDataset(Dataset):
    """CrowdPose dataset for single-person keypoint regression."""

    def __init__(self, data_root: Path, split: str = "train", transform=None, max_samples: int | None = None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.annotations = self._load_annotations()
        self.num_keypoints = 14

        if max_samples is not None and max_samples < len(self.annotations):
            self.annotations = self.annotations[:max_samples]

        self.images = {image["id"]: image for image in self.annotations_data["images"]}

    def _load_annotations(self) -> list[dict]:
        ann_path = self.data_root / "annotations" / f"crowdpose_{self.split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"CrowdPose annotation file not found: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations_data = json.load(f)

        return list(self.annotations_data["annotations"])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        ann = self.annotations[index]
        image_info = self.images[ann["image_id"]]
        image_path = self.data_root / "images" / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        # --- ORIGINAL SIZE ---
        orig_w, orig_h = image.size  # PIL gives (width, height)

        # --- LOAD KEYPOINTS ---
        keypoints = np.array(ann["keypoints"]).reshape(self.num_keypoints, 3)
        coords = keypoints[:, :2].astype(np.float32)
        visible = (keypoints[:, 2] > 0).astype(np.float32)

        # --- APPLY TRANSFORM ---
        if self.transform is not None:
            image = self.transform(image)

        # --- NEW SIZE ---
        # assumes Resize((img_size, img_size))
        new_h, new_w = image.shape[1], image.shape[2]

        # --- SCALE KEYPOINTS ---
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y

        # --- TO TENSORS ---
        coords = torch.from_numpy(coords).view(self.num_keypoints * 2)
        visible = torch.from_numpy(visible).view(self.num_keypoints)

        return image, coords, visible


class RawPoseCNN(nn.Module):
    def __init__(self, num_keypoints: int = 14, pretrained: bool = False):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_keypoints * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def smooth_l1_loss(preds: torch.Tensor, targets: torch.Tensor, visible: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Smooth L1 loss weighted by keypoint visibility for pose estimation."""
    batch_size = preds.shape[0]
    preds = preds.view(batch_size, -1, 2)
    targets = targets.view(batch_size, -1, 2)
    visible = visible.view(batch_size, -1, 1)

    diff = torch.abs(preds - targets)
    smooth_l1 = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    )
    smooth_l1 = smooth_l1 * visible
    total_visible = visible.sum()
    if total_visible < 1:
        return smooth_l1.mean()
    return smooth_l1.sum() / total_visible


def compute_oks(preds: np.ndarray, targets: np.ndarray, visible: np.ndarray, 
                bbox: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """
    Compute Object Keypoint Similarity (OKS) metric.
    
    preds, targets: (K, 2) keypoint coordinates
    visible: (K,) visibility flags
    bbox: (4,) bounding box [x, y, w, h] or None
    
    Returns (OKS score in [0, 1], per-keypoint OKS scores).
    """
    K = preds.shape[0]
    # CrowdPose keypoint per-keypoint standards (from COCO)
    sigmas = np.array([
        0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87
    ])
    
    if bbox is None:
        # If no bbox provided, use bounding box of visible keypoints
        vis_pts = targets[visible > 0]
        if len(vis_pts) == 0:
            return 0.0, np.zeros(K)
        x_min, y_min = vis_pts.min(axis=0)
        x_max, y_max = vis_pts.max(axis=0)
        bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
    
    scale = bbox[2] * bbox[3]  # bbox area
    
    if scale < 1e-6:
        return 0.0, np.zeros(K)
    
    oks_scores = []
    per_kpt_oks = np.zeros(K)
    for k in range(K):
        if visible[k] == 0:
            per_kpt_oks[k] = 0.0
            continue
        dx = preds[k, 0] - targets[k, 0]
        dy = preds[k, 1] - targets[k, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        oks = np.exp(-(dist ** 2) / (2 * (sigmas[k] ** 2) * scale))
        oks_scores.append(oks)
        per_kpt_oks[k] = oks
    
    mean_oks = float(np.mean(oks_scores)) if oks_scores else 0.0
    return mean_oks, per_kpt_oks


def compute_ap(all_oks_scores: list[float], threshold: float = 0.5) -> float:
    """Compute Average Precision at a given OKS threshold."""
    if not all_oks_scores:
        return 0.0
    all_oks_scores = np.array(all_oks_scores)
    tp = (all_oks_scores >= threshold).astype(np.float32)
    return float(tp.mean())


def compute_aphard(all_visible_oks_scores: list[float], threshold: float = 0.5) -> float:
    """
    Compute average OKS for hard keypoints (visible keypoints with OKS < threshold).
    This measures model performance on difficult-to-predict keypoints.
    """
    if not all_visible_oks_scores:
        return 0.0
    
    # Hard keypoints are those with OKS below the threshold
    hard_scores = [oks for oks in all_visible_oks_scores if oks < threshold]
    
    if not hard_scores:
        return 0.0  # No hard keypoints found, or all keypoints are easy
    
    # Return average OKS of hard keypoints
    return float(np.mean(hard_scores))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    all_oks_scores = []
    all_visible_oks_scores = []  # For AP Hard calculation
    
    with torch.no_grad():
        for images, targets, visible in loader:
            images = images.to(device)
            targets = targets.to(device)
            visible = visible.to(device)

            preds = model(images)
            loss = smooth_l1_loss(preds, targets, visible)
            loss_sum += loss.item() * images.size(0)
            
            # Compute OKS for each sample
            batch_size = preds.shape[0]
            preds_np = preds.view(batch_size, -1, 2).cpu().numpy()
            targets_np = targets.view(batch_size, -1, 2).cpu().numpy()
            visible_np = visible.view(batch_size, -1).cpu().numpy()
            
            for b in range(batch_size):
                oks, per_kpt_oks = compute_oks(preds_np[b], targets_np[b], visible_np[b])
                all_oks_scores.append(oks)
                
                # Collect OKS scores for all visible keypoints (for proper AP Hard)
                for k in range(len(visible_np[b])):
                    if visible_np[b, k] > 0:  # only visible keypoints
                        all_visible_oks_scores.append(per_kpt_oks[k])
    
    ap50 = compute_ap(all_oks_scores, threshold=0.5)
    ap75 = compute_ap(all_oks_scores, threshold=0.75)
    aphard = compute_aphard(all_visible_oks_scores, threshold=0.5)
    
    return {
        "test_loss": loss_sum / max(len(all_oks_scores), 1),
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "APHard": aphard,
        "mOKS": float(np.mean(all_oks_scores)) if all_oks_scores else 0.0,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
) -> dict[str, list]:
    metrics = {"train_loss": [], "test_loss": [], "AP@0.5": [], "AP@0.75": [], "APHard": [], "mOKS": []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, visible in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            visible = visible.to(device)

            preds = model(images)
            loss = criterion(preds, targets, visible)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics["train_loss"].append(epoch_loss)

        val_stats = evaluate(model, val_loader, device)
        metrics["test_loss"].append(val_stats["test_loss"])
        metrics["AP@0.5"].append(val_stats["AP@0.5"])
        metrics["AP@0.75"].append(val_stats["AP@0.75"])
        metrics["APHard"].append(val_stats["APHard"])
        metrics["mOKS"].append(val_stats["mOKS"])

        print(
            f"Epoch {epoch}/{epochs} - train_loss: {epoch_loss:.6f} - "
            f"test_loss: {val_stats['test_loss']:.6f} - "
            f"AP@0.5: {val_stats['AP@0.5']:.4f} - AP@0.75: {val_stats['AP@0.75']:.4f} - "
            f"APHard: {val_stats['APHard']:.4f} - mOKS: {val_stats['mOKS']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a raw CNN on a CrowdPose subset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/crowdpose"),
        help="Path to the CrowdPose data root containing images/ and annotations/",
    )
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of each split to use.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoint and metrics are saved.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="raw_cnn_crowdpose.pth",
        help="Checkpoint file name.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Image resize size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling.")
    return parser.parse_args()


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.data_root.exists():
        raise FileNotFoundError(f"CrowdPose data root not found: {args.data_root}")

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_train = CrowdPoseRawDataset(args.data_root, split="train", transform=transform)
    full_val   = CrowdPoseRawDataset(args.data_root, split="val", transform=transform)
    full_test  = CrowdPoseRawDataset(args.data_root, split="test", transform=transform)

    train_size = max(int(len(full_train) * args.fraction), 1)
    val_size   = max(int(len(full_val) * args.fraction), 1)
    test_size  = max(int(len(full_test) * args.fraction), 1)

    train_indices = random.sample(range(len(full_train)), train_size)
    val_indices   = random.sample(range(len(full_val)), val_size)
    test_indices  = random.sample(range(len(full_test)), test_size)

    train_dataset = torch.utils.data.Subset(full_train, train_indices)
    val_dataset   = torch.utils.data.Subset(full_val, val_indices)
    test_dataset  = torch.utils.data.Subset(full_test, test_indices)

    train_loader = build_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader   = build_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader  = build_loader(test_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawPoseCNN(num_keypoints=14)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = smooth_l1_loss

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    metrics_path = args.checkpoint_dir / "raw_cnn_crowdpose_metrics.json"

    print(
        f"Training on {train_size} samples, testing on {test_size} samples, "
        f"device={device}, checkpoint={checkpoint_path}"
    )

    metrics = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        args.epochs,
        checkpoint_path,
    )

    print("\nEvaluating on test set...")
    test_stats = evaluate(model, test_loader, device)

    print(
        f"TEST RESULTS - "
        f"loss: {test_stats['test_loss']:.6f}, "
        f"AP@0.5: {test_stats['AP@0.5']:.4f}, "
        f"AP@0.75: {test_stats['AP@0.75']:.4f}, "
        f"APHard: {test_stats['APHard']:.4f}, "
        f"mOKS: {test_stats['mOKS']:.4f}"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training finished. Metrics saved to {metrics_path}")
    print(f"Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
