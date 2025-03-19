from utils.logger import logger
import torch
import os


def is_good_model_with_reasonable_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.critical(
                f"name: {name}, weight: sum: {param.sum()}, mean: {param.mean()}, max: {param.max()}, min: {param.min()}"
            )
            if torch.isnan(param).any() or torch.isinf(param).any():
                return False
    return True


def load_ckpt(model, ckpt_path, optimizer, use_scheduler, scheduler):
    """
    从指定路径加载检查点文件，恢复模型和优化器的状态。

    :param model: 要恢复状态的模型
    :param ckpt_path: 检查点文件的路径
    :param optimizer: 要恢复状态的优化器
    :param use_scheduler: 是否使用学习率调度器的标志
    :param scheduler: 学习率调度器实例
    :return: 恢复的起始轮次、最佳损失值、恢复状态后的模型、恢复状态后的优化器、训练损失列表、验证损失列表以及恢复状态后的调度器
    """
    checkpoint_path = ckpt_path
    if checkpoint_path is not None and not checkpoint_path.endswith(".pth"):
        checkpoint_path = os.path.join(ckpt_path, "checkpoint_latest.pth")
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.warning("No checkpoint found at '{}'".format(checkpoint_path))
        return None

    logger.info("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    if use_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    logger.info(
        "Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, start_epoch)
    )
    return start_epoch, best_loss, model, optimizer, train_loss, val_loss, scheduler


def save_ckpt(
    work_dir,
    epoch,
    best_loss,
    model,
    optimizer,
    train_loss,
    val_loss,
    use_scheduler,
    scheduler,
):
    """
    保存模型的检查点文件。

    :param work_dir: 检查点文件保存的目录路径
    :param epoch: 当前训练的轮次
    :param best_loss: 目前为止的最佳损失值
    :param model: 要保存状态的模型
    :param optimizer: 要保存状态的优化器
    :param train_loss: 训练损失列表
    :param val_loss: 验证损失列表
    :param use_scheduler: 是否使用学习率调度器的标志
    :param scheduler: 学习率调度器实例
    :return: 如果模型状态正常并成功保存检查点，返回 True；否则返回 False
    """
    if not is_good_model_with_reasonable_weights(model):
        logger.error("model is not good, not saving checkpoint")
        return False
    checkpoint_path = os.path.join(work_dir, "checkpoint_latest.pth")
    logger.info("saving checkpoint...")
    checkpoint = {
        "epoch": epoch + 1,
        "best_loss": best_loss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if use_scheduler:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)
    # save backup
    backup_path = os.path.join(work_dir, "checkpoint_{}.pth".format(epoch + 1))
    torch.save(checkpoint, backup_path)
    # logger.info("saved checkpoint to '{}'".format(checkpoint_path))
    return True
