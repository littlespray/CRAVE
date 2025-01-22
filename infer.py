import torch
import torch.nn as nn

from models import CRAVEModel
from preprocess import Processor
import yaml
import argparse
import random
import numpy as np
from glob import glob
from tqdm.contrib import tzip
# 固定随机种子等操作
seed_n = 42
print('seed is ' + str(seed_n))
g = torch.Generator()
g.manual_seed(seed_n)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.use_deterministic_algorithms(True)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = str(seed_n)  # 为了禁止hash随机化，使得实验可复现。
config_map={
    'tradition':'configs/crave-dover.yml',
    'motion':'configs/crave_motion_aware.yml',
    'text':'configs/crave_blip_weight_3DConv_v2.yml',
    'text-local':'configs/crave_blip_local.yml',
    'text-global':'configs/crave_blip_global.yml',
    'text-individual':'configs/crave_blip_fine_grained.yml',
    # 'blip-cross':'crave_blip_cross.yml'
}


device='cuda'
class CRAVE_eva_Model(nn.Module):
    def __init__(self,key):
        super().__init__()
        with open(config_map[key], "r") as f:
            opt = yaml.safe_load(f)
        self.model = CRAVEModel(key).cuda()
        self.processor=Processor(opt['data']['videoQA']['args'])

    
    def read_data(self, path):
        branch_data=self.processor.preprocess(path)
        data={}
        for key in branch_data.keys():
                data[key]=branch_data[key]
        return data
    
    
    @torch.no_grad()
    def evaluate(self,path,prompt):
        video = self.read_data(path)
        result = self.model(video, prompt)
        return result


from VGenEval import load_prompt
def get_prompts(video_list):
    model_name = os.path.basename(video_list[0]).split('_')[0]
    id_list = []
    for video in video_list:
        num = int(os.path.basename(video).split('_')[1][:-4])
        id_list.append(num)
    prompts = load_prompt.get_prompts(id_list, model_name)['text prompt']

    return prompts


dataset_base="/dataset/new_models/new_models/" # DATASET ROOT


if __name__ == "__main__":
    t2v_videos = glob(dataset_base + "*/t2v/*.mp4")
    t2v_prompts = get_prompts(t2v_videos)
    i2v_videos = glob(dataset_base + "*/i2v/*.mp4")
    i2v_prompts = get_prompts(i2v_videos)
    for branch_key in config_map.keys():
        print(f"========={branch_key}-start==========")
        t2v_score=[]
        i2v_score=[]
        ebench = CRAVE_eva_Model(branch_key)
        print(f"========={branch_key}-t2v_evaluate-start==========")
        for video,prompt in tzip(t2v_videos,t2v_prompts):
            t2v_score+=[ebench.evaluate(video,prompt)]
        print(f"========={branch_key}-t2v_write-start==========")
        with open(f"./results/{branch_key}_t2v.txt","w") as f:
            for video,score in tzip(t2v_videos,t2v_score):
                f.write(f"{video.split('/')[-1]},{score.item()}\n")
        print(f"========={branch_key}-i2v_evaluate-start==========")
        for video,prompt in tzip(i2v_videos,i2v_prompts):
            i2v_score+=[ebench.evaluate(video,prompt)]
        print(f"========={branch_key}-t2v_write-start==========")
        with open(f"./results/{branch_key}_i2v.txt","w") as f:
            for video,score in tzip(i2v_videos,i2v_score):
                f.write(f"{video.split('/')[-1]},{score.item()}\n")
        del ebench
