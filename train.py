import pickle
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
import random
import torch.cuda as cuda
import os
import sys

sys.path.insert(0, "./")

from datasets import mot15_curve as datasets
from torch.utils.data import DataLoader

from model.SHENet import SHENet
from utils.loss_funcs import CurveLoss
from utils.data_utils import define_actions
from utils.parser import args
from utils.logger import logger, init_logger
from utils.model_utils import load_ckpt, save_ckpt
import datetime
from tqdm import tqdm
import pudb
import wandb
import psutil


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# torch.autograd.set_detect_anomaly(True)


def set_random_seed(seed: int):
    if seed == -1:
        seed = random.randint(0, 99999)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    return seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s" % device)

# PETS
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
work_dir = os.path.join(args.model_path, date_str)
init_logger(work_dir)


def train(model, resume_ckpt_path=None):
    wandb.init(
        project="SHENet",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "architecture": "SHENet",
            "timestamp": date_str,
        },
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )

    train_loss = []
    val_loss = []

    train_sequences = ["PETS09-S2L1"]
    val_sequences = ["PETS09-S2L1"]

    dataset = datasets.MOT(args, sequences=train_sequences, split=0)
    logger.info(">>> Training dataset length: {:d}".format(dataset.__len__()))
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    dataset_length = len(data_loader)

    vald_dataset = datasets.MOT(args, sequences=val_sequences, split=1)
    logger.info(">>> Validation dataset length: {:d}".format(vald_dataset.__len__()))
    vald_loader = DataLoader(
        vald_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    best_loss = 1000
    start_epoch = 0
    cur_loss = -1

    # 检查是否存在检查点文件
    train_info = load_ckpt(
        model, resume_ckpt_path, optimizer, args.use_scheduler, scheduler
    )
    if train_info is not None:
        start_epoch, best_loss, model, optimizer, train_loss, val_loss, scheduler = (
            train_info
        )
    wandb.watch(model, log="all", log_freq=1)

    # 定义参数
    static_memory = model.load_static_memory(
        "/home/alvin.gao/SHENet/data/SHENet/pretrained/FinalBank.pt"
    )
    criterion = CurveLoss(static_memory, args.memory_size)

    for epoch in range(start_epoch, args.n_epochs):
        running_loss = 0
        n = 0
        model.train()

        all_trajs = []
        for cnt, (input_root, target, scale, meta, raw_img) in enumerate(data_loader):
            if args.save_trajectories:
                trajs = [
                    input_root[i].cpu().numpy() for i in range(input_root.shape[0])
                ]
                all_trajs.extend(trajs)
                continue
            batch_dim = input_root.shape[0]
            n += batch_dim
            input_root = input_root.float().cuda()
            target = target.float().cuda()
            scale = scale.cuda()
            raw_img = raw_img.cuda()

            preds = model(input_root[:, : args.input_n], raw_img)

            loss, _ = criterion(preds, input_root, target, False)

            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss / (1024.0 * 1024.0)
            iteration = epoch * dataset_length + cnt
            anchor_trajs_count = len(criterion.memory_curves)
            logger.info(
                f"iter: {iteration} \t \t [{epoch + 1}, {cnt + 1}]  training loss: {loss.item()}, anchor_trajs_count: {anchor_trajs_count}, mem: {cpu_memory:.2f}MB"
            )
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "memory": cpu_memory,
                    "anchor_trajs": anchor_trajs_count,
                    "epoch": epoch,
                },
                step=iteration,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.detach() * batch_dim

        if args.save_trajectories:
            logger.info("saving trajectories...")
            traj_to_save = [{"root": i} for i in all_trajs]
            torch.save(traj_to_save, "./trajs.pt")
            logger.critical("trajectories saved! exit...")
            exit(0)

        if running_loss == 0:
            logger.error("running_loss is 0")
            continue
        train_loss.append(running_loss.detach().cpu() / n)
        model.eval()

        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, (input_root, target, scale, meta, raw_img) in enumerate(
                vald_loader
            ):
                batch_dim = input_root.shape[0]
                n += batch_dim
                input_root = input_root.float().cuda()
                target = target.float().cuda()
                scale = scale.cuda()

                raw_img = raw_img.cuda()

                preds = model(input_root[:, : args.input_n], raw_img)

                loss, _ = criterion(preds, input_root, target, False)

                if cnt % 500 == 0:
                    logger.info(
                        "[%d, %5d]  validation loss: %.3f"
                        % (epoch + 1, cnt + 1, loss.item())
                    )
                running_loss += loss * batch_dim
            cur_loss = running_loss.detach().cpu() / n
            val_loss.append(cur_loss)
        dynamic_memory = criterion.memory_curves
        logger.info(f"dynamic memory size: {len(dynamic_memory)}")

        if args.use_scheduler:
            scheduler.step()
        if best_loss > cur_loss:
            best_loss = cur_loss
            logger.info("Epoch %d, best loss: %.3f" % (epoch + 1, best_loss))
            torch.save(dynamic_memory, os.path.join(work_dir, "mem_curves_bank.pt"))
            torch.save(model.state_dict(), os.path.join(work_dir, "model.pth"))

        # 保存检查点
        if not save_ckpt(
            work_dir,
            epoch,
            best_loss,
            model,
            optimizer,
            train_loss,
            val_loss,
            args.use_scheduler,
            scheduler,
        ):
            raise Exception("model is not good, not saving checkpoint")

    pickle_out = open(work_dir + "val_loss.pkl", "wb")
    pickle.dump(val_loss, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    model = SHENet(args)
    model = model.cuda()
    resume_model_path = None
    logger.info(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    train(model, resume_model_path)
