# python -m debugpy --listen 5678 train_ctl_model.py \
python train_ctl_model_cross.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.DEST 'dukemtmcreid' \
DATASETS.ROOT_DIR './data' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs_test/market1501/256_resnet50/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/hgst/projects/personreid/centroids-reid/logs-1.0-5.0/market1501/64_0.2_resnet50/17092024_222356/default/auto_checkpoints/checkpoint_119.pth" \
MODEL.USE_CENTROIDS True \
MODEL.KEEP_CAMID_CENTROIDS True \
REPRODUCIBLE_NUM_RUNS 1