# python -m debugpy --listen 5678 train_ctl_model.py \
python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs_test/market1501/256_resnet50/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/hgst/projects/personreid/centroids-reid/A100/logs/market1501/True_False_1.0_False_1.0_False_0.0005_False_1.0_True_1.0_64_resnet50/21092024_154318/default/checkpoints/epoch=39.ckpt" \
MODEL.USE_CENTROIDS True \
MODEL.KEEP_CAMID_CENTROIDS True