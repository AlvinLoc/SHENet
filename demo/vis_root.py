from cv2 import error
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def vis_img(
    img_path,
    person_id,
    start_frame,
    input_n,
    root_pred,
    root_gt,
    root_smooth_gt,
    anchor_traj,
    save_dir,
):
    img = imread(img_path)

    n = root_gt.shape[0]
    m = root_pred.shape[0]
    assert n == m, f"gt and pred length not equal, gt: {n}, pred: {m}"

    height, width = img.shape[:2]

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.imshow(img)

    plt.plot(
        root_gt[:, 0],
        root_gt[:, 1],
        "g",
        linewidth=2,
        marker="o",
        markersize=4,
        zorder=10,
    )

    plt.plot(
        root_pred[:, 0],
        root_pred[:, 1],
        "r",
        linewidth=2,
        marker="o",
        markersize=4,
        zorder=11,
    )
    plt.plot(
        root_smooth_gt[:, 0],
        root_smooth_gt[:, 1],
        "y",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        anchor_traj[:, 0],
        anchor_traj[:, 1],
        "m",
        linewidth=1,
        marker="o",
        markersize=4,
    )

    plt.scatter(
        root_gt[input_n - 1, 0],
        root_gt[input_n - 1, 1],
        c="y",
        s=100,
        marker="*",
        zorder=10,
    )
    plt.scatter(
        root_pred[input_n - 1, 0],
        root_pred[input_n - 1, 1],
        c="b",
        s=100,
        marker="*",
        zorder=11,
    )
    plt.scatter(
        anchor_traj[input_n - 1, 0],
        anchor_traj[input_n - 1, 1],
        c="m",
        s=50,
        marker="*",
        zorder=11,
    )

    plt.text(5, 50, "pred trajectory", color="r", fontsize=12)
    plt.text(10, 100, "gt trajectory", color="g", fontsize=12)
    plt.text(
        15,
        150,
        f"{img_path}",
    )
    plt.text(
        20,
        200,
        f"person_id: {person_id}",
    )
    plt.text(
        25,
        250,
        f"start_frame: {start_frame}",
    )

    plt.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    sub_dir_path = os.path.join(save_dir, str(person_id))
    os.makedirs(sub_dir_path, exist_ok=True)

    base_name = os.path.basename(img_path)
    save_path = os.path.join(sub_dir_path, base_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def run(joint, idx, input_n, output_n, save_path):
    root_pred = np.array(joint[idx]["pred_root"])
    root_gt = np.array(joint[idx]["root"])
    root_smooth_gt = np.array(joint[idx]["smooth_root"])
    anchor_traj = np.array(joint[idx]["anchor_traj"])
    img_path = joint[idx]["img_path"]
    person_id = joint[idx]["person_id"]
    start_frame = joint[idx]["start_frame"]

    gt_start_pt = root_gt[0, :]
    pred_start_pt = root_pred[0, :]
    diff = gt_start_pt - pred_start_pt
    max_diff = max(abs(diff.min()), abs(diff.max()))
    assert max_diff < 0.1, f"max start point dist: {max_diff}"

    vis_img(
        img_path,
        person_id,
        start_frame,
        input_n,
        root_pred,
        root_gt,
        root_smooth_gt,
        anchor_traj,
        save_path,
    )


if __name__ == "__main__":
    # lstm offsets classifys_offsets  scene1_curve, lstm_curve
    work_dir = "output/test_2025-03-30_16-09-16"
    venice_path = os.path.join(work_dir, "venice.json")
    with open(venice_path, "r") as f:
        data = json.load(f)

    root_gt = []
    save_path = os.path.join(work_dir, "vis_root")
    os.makedirs(save_path, exist_ok=True)

    input_n = 10
    output_n = 50
    for idx in tqdm(range(1, len(data), 1)):
        run(data, idx, input_n, output_n, save_path)
