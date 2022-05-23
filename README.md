# CLIMS

Code repository for our paper "[CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02668)" in **CVPR 2022**.

![](clims.png)

Please to **NOTE** that this repository is an **improved version** of our camera-ready version (you can refer to the directory of `previous_version/`). We recommend to use our improved version of CLIMS instead of camera-ready version.

## Dataset
### PASCAL VOC2012
You will need to download the images (JPEG format) in PASCAL VOC2012 dataset at here. Make sure your `data/VOC2012 folder` is structured as follows:
```
├── VOC2012/
|   ├── Annotations
|   ├── ImageSets
|   ├── SegmentationClass
|   ├── SegmentationClassAug
|   └── SegmentationObject
```
### MS-COCO 2014 (coming soon) 

## Training
1. Download pre-trained baseline CAM ('res50_cam.pth') at [here](https://drive.google.com/drive/folders/1CCYduc2L_V_s7MtXEuA_LzIscdlFFJag?usp=sharing) and put it at the directory of `cam-baseline-voc12/`
2. Train CLIMS on PASCAL V0C2012 dataset to generate initial CAMs
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root /data1/xjheng/dataset/VOC2012/ --hyper 10,24,1,0.2 --clims_num_epoches 15 --cam_eval_thres 0.15 --work_space clims_voc12 --cam_network net.resnet50_clims --train_clims_pass True --make_clims_pass True --eval_cam_pass True
```
2. Train IRNet and generate pseudo semantic masks
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root /data1/xjheng/dataset/VOC2012/ --cam_eval_thres 0.15 --work_space clims_voc12 --cam_network net.resnet50_clims --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True
```
3. Train DeepLabv2 using pseudo semantic masks. (Please refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch))

## Evaluation Results
### The quality of initial CAMs and pseudo masks on PASCAL VOC2012.

| Method    | backbone | CAMs | + RW | + IRNet |
|:---------:|:--------:|:----:|:----:|:----:|
| **CLIMS(cvpr)** | R50      | 56.6 | 70.5 | - |
| **CLIMS(this repo)**    | R50      | 58.6 | ~73 | 74.1 |

### Evaluation results on PASCAL VOC2012 val and test sets.

| Method    | Supervision | Network  | Pretrained  | val  | test |
|:---------:|:-----------:|:----:|:----:|:----:|:----:|
| AdvCAM    | S           | DeepLabV2 |  ImageNet | 68.1 | 68.0 |
| EDAM      | I+S         | DeepLabV2 |  COCO     | 70.9 | 70.6 |
| **CLIMS(cvpr)** | I     | DeepLabV2 |  ImageNet | 69.3 | 68.7 |
| **CLIMS(cvpr)** | I     | DeepLabV2 |  COCO     | 70.4 | 70.0 |
| **CLIMS(this repo)** | I     | DeepLabV2 |ImageNet | 70.3 | 70.6 |
| **CLIMS(this repo)** | I     | DeepLabV2 | COCO     | 71.4 | 71.2 |
| **CLIMS(this repo)** | I     | DeepLabV1-R38 | COCO     | 73.3 | 73.4 |

(Initial CAMs, pseudo semantic masks, and pre-trained models of camera-ready version can be found at [Google Drive](https://drive.google.com/drive/folders/1njCaolWacqSmw7HVNecwvCAMm7NsCFPq?usp=sharing))

If you are using our code, please consider citing our paper.

```
@article{xie2022cross,
  title={Cross Language Image Matching for Weakly Supervised Semantic Segmentation},
  author={Xie, Jinheng and Hou, Xianxu and Ye, Kai and Shen, Linlin},
  journal={arXiv preprint arXiv:2203.02668},
  year={2022}
}
```
This repository was highly based on [IRNet](https://github.com/jiwoon-ahn/irn), thanks for Jiwoon Ahn's great code.