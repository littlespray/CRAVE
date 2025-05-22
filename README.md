# CRAVE - Content-Rich AIGC Video Quality Assessment via Intricate Text Alignment and Motion-Aware Consistency

## Introduction
Current video models such as Sora offer substantial improvement in generation quality compared to previous models, characterized by rich details and content. These models support longer text control (often over 200 characters) and longer duration (often over 5 seconds with the fps of 24). Compared with previous AIGC videos, these videos rarely encounter flicker issues that were commonly seen in previous models, and usually have more intricate prompts, more complex motion patterns and richer details.  To evaluate the new generation of video generation models, we introduced CRAVE.


CRAVE evaluates AIGC video quality from three perspectives: traditional natural video quality assessment angles such as aesthetics and distortion,
text-video semantic alignment via Multi-granularity TextTemporal (MTT) fusion, and the specific dynamic distortions in AIGC videos via Sparse-Dense Motion-aware
(SDM) modules. The overall framework is illustrated in Figure 3. For traditional natural video quality assessment, we
utilize Dover to assess individual videos from the aesthetic and technical perspectives given its success. For other aspects, we design effective sparse-dense
motion-aware video dynamics modeling, and multi-granularity text-temporal fusion module that aligns textual semantics with complex spatio-temporal relationships in video clips.


## Preparation

### Environment Setup
```
pip install -r requirements.txt
```

### Pretrained Models
Download pre-trained models [here](https://drive.google.com/drive/folders/1DTHEW3pGS_6mLO1PvnXz4k0_sf3r9Oww?usp=sharing) and put them into ``ckpts``.

Download model weights from [google drive](https://drive.google.com/file/d/1TRgQjD6stSf3OMMiiopCq7MzlgUcKG5x/view?usp=sharing) / [baidu yun](https://pan.baidu.com/s/1X8sTTriwyzgMBqAjb8r2NQ?pwd=v5rv) and then put them into ``pretrained_weights``.

## Usage
To run the demo in the code, please additionally install [VideoGenEval](https://github.com/AILab-CVC/VideoGen-Eval) and prepare related data.
```
python infer.py
```

## Acknowledgement
This repo is built on [BLIP](https://github.com/salesforce/BLIP) and [DOVER](https://github.com/VQAssessment/DOVER). We thank the authors for their nice work.

