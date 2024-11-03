import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

from train_ctl_model import CTLModel

### Prepare logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

### Functions used to extract pair_id
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001
exctract_func = lambda x: Path(
    x
).parent.name  ## To extract pid from parent directory of an iamge. Example: /path/to/root/001/image_04.jpg -> pid = 001

if __name__ == "__main__":
    ### Build model
    model = CTLModel.load_from_checkpoint("/hgst/projects/personreid/centroids-reid/logs-1.0-5.0/market1501/64_0.1_resnet50/17092024_222356/default/auto_checkpoints/checkpoint_119.pth")
    
    # Get the underlying PyTorch model
    pytorch_model = model.model  # This assumes that your `MyLightningModel` has a model attribute.
    
    # Save the PyTorch model's state dict
    torch.save(pytorch_model.state_dict(), "best_model.pth")

    ### Inference
    log.info("Running inference")

    ### Create centroids
    ### Save
