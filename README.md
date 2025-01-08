# SCM-ReID

Code Repo for SCM-ReID method.

The codes are expanded on a [[centroids]](https://github.com/mikwieczorek/centroids-reid) and [[SupCon]](https://github.com/HobbitLong/SupContrast).

## Get Started

The whole model is implemented in PyTorch-Lightning framework.

1. `cd` to a directory where you want to clone this repo
2. Run `git clone https://github.com/dangph258/scm-reid.git`
3. Install conda enviroment `conda env create -f scm-reid.yml`
4. Download pre-trained weights into `models/` directory for:
    - Resnet50 from here: [[link]](https://download.pytorch.org/models/resnet50-19c8e357.pth)
5. Prepare datasets:

    Market1501
    * Extract dataset and rename to `market1501` inside `data/`
    * The data structure should be following:

    ```bash
    /data
        market1501
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    CUHK03

    * Extract dataset and rename to `cuhk03` inside data/
    * The data structure should be following:

    ```bash
    /data
        cuhk03
           	cuhk03_release/
           	splits_classic_detected.json
            splits_classic_labeled.json
           	......
    ```

## Train
Each Dataset and Model has its own train script.  
All train scripts are in `train_scirpts` folder with corresponding dataset name.

Example run command to train SCM-ReID on market1501
```bash
CUDA_VISIBLE_DEVICES=0 ./train_scripts/market1501/md_train_ctl_model_s_r50_market1501.sh
```


## Test
To test the trained model you can use provided scripts in `train_scripts`, just two parameters need to be added:  
    
    TEST.ONLY_TEST True \  
    MODEL.PRETRAIN_PATH "path/to/pretrained/model/checkpoint.pth"
    
Example train script for testing trained SCL-ReID on Market1501
```bash
python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "logs/market1501/exp/backbone/time/default/checkpoints/epoch=119.ckpt"
```
