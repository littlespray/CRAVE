from .backbone.uniformer_backbone import uniformerv2_b16
from .backbone.StreamFlowT4 import StreamFlowT4
from .head import VQAHead_cls,VARHead,VQAHead
import torch.nn as nn
import torch
from functools import reduce

class DoubleStreamModel(nn.Module):
    def __init__(
            self,
            backbone_size="divided",
            backbone_preserve_keys="fragments,resize",
            multi=False,
            layer=-1,
            backbone=dict(
                resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
            ),
            divide_head=False,
            head_type='VQAhead_cls',
            vqa_head=dict(in_channels=768),
            var=False,
            use_tn=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()

        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "uniformerv2_b16":
                b = uniformerv2_b16(temporal_downsample=False, no_lmhra=True, t_size=32)
            elif t_backbone_size == "StreamFlowT4":
                b = StreamFlowT4()
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
        if divide_head:
            for key in backbone:
                pre_pool = False  # if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                if key =="time":
                    in_channel = 128
                else:
                    in_channel=768
                b = VQAHead_cls(pre_pool=pre_pool, in_channels=in_channel, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)

    def forward(
            self,
            vclips,
            prompts=None,
            inference=True,
            return_pooled_feats=False,
            return_raw_feats=False,
            reduce_scores=False,
            pooled=False,
            **kwargs
    ):
        # import pdb;pdb.set_trace()
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = []
                for key in self.backbone_preserve_keys:
                        if "time" in key:
                            feat = getattr(self, key.split("_")[0] + "_backbone")(
                                vclips[key]#, prompts
                            )
                        else:
                            feat = getattr(self, key.split("_")[0] + "_backbone")(
                                vclips[key]
                            )
                        if hasattr(self, key.split("_")[0] + "_head"):
                            scores += [getattr(self, key.split("_")[0] + "_head")(feat)[0]]
                        else:
                            scores += [getattr(self, "vqa_head")(feat)]

                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
            #self.train()
            if return_pooled_feats or return_raw_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = []
            for key in vclips:
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                            vclips[key]
                    )
                    feats.append(feat)
            feats = (torch.cat(feats, dim=1))
            if hasattr(self, key.split("_")[0] + "_head"):
                scores += [getattr(self, key.split("_")[0] + "_head")(feats)[0]]
            else:
                scores += [getattr(self, "vqa_head")(feats)]
            scores += [torch.zeros_like(scores[0])]
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    # print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    # print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores