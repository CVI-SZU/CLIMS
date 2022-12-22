## DeepLabv2
This repository was highly based on [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).
### Step 1
Please download pre-trained weights from [Here](https://drive.google.com/drive/folders/1nsXWLoK1w56iC9DE5jwdcqQDX8of4DH5?usp=share_link) and put it into the directory `weights/`.
### Step 2
Please specify the data root and experiment name in `config/xxx.yaml`.
### Step 3
```shell
CUDA_VISIBLE_DEVICES=0 bash run_voc12_coco_pretrained.sh
CUDA_VISIBLE_DEVICES=0 bash run_voc12_imagenet_pretrained.sh
```
### Step 4 - Submit evaluation results to PASCAL VOC [evaluation server](http://host.robots.ox.ac.uk:8080/).
Please check the directory `data/features/your/experiment/name/deeplabv2_resnet101_msc/test/`. Please submit `results.tar.gz` to the server.
