import yaml

import torch
import torch.nn as nn

from .tradition import DOVER
from .fidelity import DoubleStreamModel
from .text_alignment import VideoTextAlignmentModel

config_map={
    'tradition':'configs/crave-dover.yml',
    'motion':'configs/crave_motion_aware.yml',
    'text':'configs/crave_blip_weight_3DConv_v2.yml',
    'text-local':'configs/crave_blip_local.yml',
    'text-global':'configs/crave_blip_global.yml',
    'text-individual':'configs/crave_blip_fine_grained.yml',
    # 'blip-cross':'crave_blip_cross.yml'
}


class CRAVEModel(nn.Module):
    def __init__(self,key):
        super().__init__()
        self.key=key
        with open(config_map[key], "r") as f:
            opt = yaml.safe_load(f)
        # build model
        if 'tradition' in key:
            self.branch = DOVER(**opt['model']['args']).eval()
        elif 'motion' in key:
            self.branch = DoubleStreamModel(**opt['model']['args']).eval()
        elif 'text' in key:
            self.branch = VideoTextAlignmentModel(**opt['model']['args']).eval()
        ckpts = opt['test_load_path']
        self.load_ckpt(ckpts)

    # TODO
    def load_ckpt(self, ckpt_folder):
        self.branch.load_state_dict(torch.load(ckpt_folder,map_location='cpu')['state_dict'],strict=False)
        self.branch.eval()
        self.eval()
    # TODO
    def forward(self, video, prompt):
        score = self.branch(video,prompt,reduce_scores=True,pooled=True)
        return score



if __name__ == "__main__":
    eval_model=CRAVEModel('tradition')
