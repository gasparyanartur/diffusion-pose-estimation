"""Generate CPU-only plots and summary data for the DiffusionRegPose report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "diffusionregpose_training_gap"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ap = pd.DataFrame(
        [
            ("Official released checkpoint", 0.726776, 0.631000, "all-test official"),
            ("External reproduction, epoch 80", 0.688012, 0.620572, "all-test external"),
            ("Native best, epoch 50", 0.700491, 0.620248, "all-test native validation"),
            ("Native final, epoch 80", 0.674500, 0.592545, "all-test native validation"),
        ],
        columns=["run", "keypoint_ap", "bbox_ap", "evaluation"],
    )
    ap.to_csv(REPORT_DIR / "ap_comparison.csv", index=False)

    native = pd.DataFrame(
        [
            (10, 0.535704, 0.507207, "reconstructed status record"),
            (20, 0.559386, 0.526618, "reconstructed status record"),
            (30, 0.592805, 0.558205, "reconstructed status record"),
            (40, 0.696976, 0.615994, "reconstructed status record"),
            (50, 0.700491, 0.620248, "reconstructed status record"),
            (60, 0.683309, 0.603909, "reconstructed status record"),
            (70, 0.671823, 0.587275, "resume CSV"),
            (80, 0.674500, 0.592545, "resume CSV"),
        ],
        columns=["epoch", "keypoint_ap", "bbox_ap", "source"],
    )
    native.to_csv(REPORT_DIR / "native_trajectory.csv", index=False)

    loss = pd.DataFrame(
        [
            (63, 17.384989), (64, 17.362518), (65, 17.358309),
            (66, 17.335426), (67, 17.379581), (68, 17.338373),
            (69, 17.345896), (70, 17.318106), (71, 17.310471),
            (72, 17.334749), (73, 17.296717), (74, 17.274170),
            (75, 17.296297), (76, 17.317801), (77, 17.316654),
            (78, 17.321392), (79, 17.276241),
        ],
        columns=["epoch", "loss"],
    )
    loss.to_csv(REPORT_DIR / "native_resume_loss.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = range(len(ap))
    width = 0.36
    axis.bar([p - width / 2 for p in positions], ap.keypoint_ap, width, label="Keypoint AP")
    axis.bar([p + width / 2 for p in positions], ap.bbox_ap, width, label="Bbox AP")
    axis.set_xticks(list(positions), ["Official\ncheckpoint", "External\n80 epoch", "Native\nbest", "Native\nfinal"])
    axis.set_ylim(0.5, 0.76)
    axis.set_ylabel("AP")
    axis.set_title("DiffusionRegPose evaluation comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ap_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(native.epoch, native.keypoint_ap, marker="o", label="Native keypoint AP")
    axis.plot(native.epoch, native.bbox_ap, marker="o", label="Native bbox AP")
    axis.axhline(0.726776, color="black", linestyle="--", label="Official keypoint AP")
    axis.axhline(0.688012, color="tab:orange", linestyle=":", label="External 80-epoch keypoint AP")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("AP")
    axis.set_ylim(0.45, 0.76)
    axis.set_title("Native training trajectory")
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "native_trajectory.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(loss.epoch, loss.loss, marker="o", color="tab:purple")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation loss")
    axis.set_title("Native resumed validation loss, epochs 63–79")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "native_resume_loss.png", dpi=180)
    plt.close(figure)

    metadata = {
        "official_checkpoint": "external/DiffusionRegPose/pretrain_models/diffusionregpose_r50_crowdpose.pth",
        "external_reproduction_checkpoint": "external/DiffusionRegPose/logs/crowdpose_r50_effective_batch8_resume80/checkpoint.pth",
        "native_run": "results/diffusionregpose_paper_full/diffusionregpose_paper_full/20260903_000016",
        "native_checkpoint": "results/diffusionregpose_paper_full/diffusionregpose_paper_full/20260903_000016/checkpoints/epoch079.ckpt",
        "native_evaluator_anchor": {"keypoint_ap": 0.726776, "bbox_ap": 0.631},
        "native_history_caveat": "Epochs 10-60 were reconstructed from recorded status artifacts after the original CSV logger was overwritten during the first resume attempt.",
    }
    (REPORT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
