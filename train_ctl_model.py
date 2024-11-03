# encoding: utf-8
"""
Adapted and extended by:
@author: mikwieczorek
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities import AttributeDict, rank_zero_only
# from torch import tensor
# from tqdm import tqdm

from config import cfg
from modelling.bases import ModelBase
from utils.misc import run_main

from utils.util import AverageMeter
import time


class CTLModel(ModelBase):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.losses_names = [
            "query_xent",
            "query_triplet",
            "query_center",
            # "query_focal",
            "centroid_triplet",
            "supcon_loss"
        ]
        self.losses_dict = {n: [] for n in self.losses_names}
        
        self.batch_time = AverageMeter()
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):  
        start = time.time()
        opt, opt_center = self.optimizers(use_pl_optimizer=True)

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if self.trainer.current_epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0,
                    float(self.trainer.current_epoch + 1)
                    / float(self.hparams.SOLVER.WARMUP_EPOCHS),
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR
                    
        opt.zero_grad()
        if self.hparams.SOLVER.USE_CENTER_LOSS:
            opt_center.zero_grad()

        x, y, class_labels, camid, isReal = batch  # batch is a tuple

        unique_classes = len(np.unique(class_labels.detach().cpu()))

        # Get backbone features
        _, features = self.backbone(x)

        # Loss: contrastive_loss_query
        contrastive_loss_query = torch.tensor(0.0)
        if self.hparams.SOLVER.USE_CONTRASTIVE_LOSS:
            contrastive_loss_query, _, _ = self.contrastive_loss(
                features, class_labels, mask=isReal
            )
            contrastive_loss_query = (
                contrastive_loss_query * self.hparams.SOLVER.CONTRASTIVE_LOSS_WEIGHT
            )

        class_labels_real = class_labels[isReal]
        features_real = features[isReal]
            
        # Loss: center_loss
        center_loss = torch.tensor(0.0)
        if self.hparams.SOLVER.USE_CENTER_LOSS:
            # class_labels_real = class_labels[isReal]
            # features_real = features[isReal]
            center_loss = self.hparams.SOLVER.CENTER_LOSS_WEIGHT * self.center_loss(
                features_real, class_labels_real
            )
        
        # Loss: xent_query
        xent_query = torch.tensor(0.0)
        # focal_query = torch.tensor(0.0)
        # if self.hparams.SOLVER.USE_XENT_LOSS or self.hparams.SOLVER.USE_FOCAL_LOSS: 
        if self.hparams.SOLVER.USE_XENT_LOSS:
            bn_features = self.bn(features_real)
            cls_score = self.fc_query(bn_features)
            if self.hparams.SOLVER.USE_XENT_LOSS:
                xent_query = self.xent(cls_score, class_labels_real)
                # xent_query = F.cross_entropy(cls_score, class_labels_real, reduction='none')
                # xent_query = xent_query.mean(0).sum()
                xent_query = xent_query * self.hparams.SOLVER.XENT_LOSS_WEIGHT
                
            # if self.hparams.SOLVER.USE_FOCAL_LOSS:
            #     focal_query = self.focal(cls_score, class_labels_real)
            #     focal_query = focal_query * self.hparams.SOLVER.FOCAL_LOSS_WEIGHT
        
        # Loss: Supcon Loss
        supcon_loss = torch.tensor(0.0)
        if self.hparams.SOLVER.USE_SUPERVISED_CONTRASTIVE_LOSS:
            images = torch.cat([x, y], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                class_labels = class_labels.cuda(non_blocking=True)
            bsz = class_labels.shape[0]
            
            # compute loss
            # features  = self.model(images,supCon=True)
            # Get backbone features
            _, features_sup = self.backbone(images)
            bn_features = self.bn(features_sup)
            fc_features = self.fc_query(bn_features)
            bn_x_sup = F.normalize(fc_features)
            
            f1, f2 = torch.split(bn_x_sup, [bsz, bsz], dim=0)
            features_sup = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            supcon_loss = self.supcon(features_sup, class_labels)
            supcon_loss = supcon_loss * self.hparams.SOLVER.SUPERVISED_CONTRASTIVE_LOSS_WEIGHT
        
        #Loss: Centroid Triploss
        # Prepare masks for uneven numbe of sample per pid in a batch
        contrastive_loss_step = torch.tensor(0.0)
        if self.hparams.SOLVER.USE_CENTROID_CONTRASTIVE_LOSS:
            ir = isReal.view(unique_classes, -1)
            t = repeat(ir, "c b -> c b s", s=self.hparams.DATALOADER.NUM_INSTANCE)
            t_re = rearrange(t, "c b s -> b (c s)")
            t_re = t_re & isReal

            masks, labels_list = self.create_masks_train(class_labels)  ## True for gallery
            masks = masks.to(features.device)
            masks = masks & t_re

            masks_float = masks.float().to(features.device)
            padded = masks_float.unsqueeze(-1) * features.unsqueeze(0)  # For broadcasting

            centroids_mask = rearrange(
                masks, "i (ins s) -> i ins s", s=self.hparams.DATALOADER.NUM_INSTANCE
            )
            padded_tmp = rearrange(
                padded,
                "i (ins s) dim -> i ins s dim",
                s=self.hparams.DATALOADER.NUM_INSTANCE,
            )
            valid_inst = centroids_mask.sum(-1)
            valid_inst_bool = centroids_mask.sum(-1).bool()
            centroids_emb = padded_tmp.sum(-2) / valid_inst.masked_fill(
                valid_inst == 0, 1
            ).unsqueeze(-1)

            contrastive_loss_total = []
            ap_total = []
            an_total = []
            l2_mean_norm_total = []
            xent_centroids_total = []

            for i in range(self.hparams.DATALOADER.NUM_INSTANCE):
                if valid_inst_bool[i].sum() <= 1:
                    continue

                current_mask = masks[i, :]
                current_labels = class_labels[~current_mask & t_re[i]]
                query_feat = features[~current_mask & t_re[i]]
                current_centroids = centroids_emb[i]
                current_centroids = current_centroids[
                    torch.abs(current_centroids).sum(1) > 1e-7
                ]
                embeddings_concat = torch.cat((query_feat, current_centroids))
                labels_concat = torch.cat((current_labels, current_labels))

                contrastive_loss, dist_ap, dist_an = self.contrastive_loss(
                    embeddings_concat, labels_concat
                )

                with torch.no_grad():
                    dist_ap = dist_ap.data.mean()
                    dist_an = dist_an.data.mean()
                ap_total.append(dist_ap)
                an_total.append(dist_an)

                contrastive_loss_total.append(contrastive_loss)

                # L2 norm of centroid vectors
                l2_mean_norm = torch.norm(current_centroids, dim=1).mean()
                l2_mean_norm_total.append(l2_mean_norm)

            contrastive_loss_step = (
                torch.mean(torch.stack(contrastive_loss_total))
                * self.hparams.SOLVER.CENTROID_CONTRASTIVE_WEIGHT
            )
            dist_ap = torch.mean(torch.stack(ap_total))
            dist_an = torch.mean(torch.stack(an_total))
            l2_mean_norm_total = torch.mean(torch.stack(l2_mean_norm_total))

        total_loss = (
            contrastive_loss_step + center_loss + xent_query + contrastive_loss_query + supcon_loss
        )

        self.manual_backward(total_loss, optimizer=opt)
        opt.step()
        if self.hparams.SOLVER.USE_CENTER_LOSS:
            for param in self.center_loss.parameters():
                param.grad.data *= 1.0 / self.hparams.SOLVER.CENTER_LOSS_WEIGHT
            opt_center.step()
            
        losses = [
            xent_query,
            contrastive_loss_query,
            center_loss,
            contrastive_loss_step,
            supcon_loss
        ]
        losses = [item.detach() for item in losses]
        losses = list(map(float, losses))
        loss_out = ""
        for name, loss_val in zip(self.losses_names, losses):
            self.losses_dict[name].append(loss_val)
            # loss_out += f'{name}: {loss_val}; '
        
        # print(loss_out)
        log_data = {
            "step_dist_ap": float(0.0),
            "step_dist_an": float(0.0),
            "l2_mean_centroid": float(0.0),
        }
        if self.hparams.SOLVER.USE_CENTROID_CONTRASTIVE_LOSS:
            log_data = {
                "step_dist_ap": float(dist_ap),
                "step_dist_an": float(dist_an),
                "l2_mean_centroid": float(l2_mean_norm_total),
            }
        self.batch_time.update(time.time() - start)

        return {"loss": total_loss, "other": log_data}
    
    def training_epoch_end(self, outputs):
        
        print('Batch time {batch_time.val:.3f} {batch_time.avg:.3f})'.format(batch_time=self.batch_time))
        
        if hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch + 1)

        lr = self.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])
        l2_mean_norm = np.mean([x["other"].pop("l2_mean_centroid") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
            "l2_mean_centroid": l2_mean_norm,
        }

        if hasattr(self, "losses_dict"):
            for name, loss_val in self.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                self.losses_dict[name] = []  ## Zeroing values after a completed epoch
                
        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)
        self.trainer.accelerator_backend.barrier()

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="CLT Model Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(cfg)
    logger_save_dir = f"{Path(__file__).stem}"
    print("---Config time: ", time.time() - start)
    run_main(cfg, CTLModel, logger_save_dir)
    print("---Running time: ", time.time() - start)
