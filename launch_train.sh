CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda python3 train.py \
    --data_dir ./data/MOT15 \
    --vit_config_file ./third-party/mmsegmentation/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py \
    --vit_checkpoint_file ./pretrained_models/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth \
    --gpus 1 
    # --gpus 1 \
    # --lr 1e-5 \
    # --n_epochs 100