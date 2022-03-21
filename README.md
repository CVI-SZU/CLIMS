# CLIMS
Code repository for our paper "[CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02668)" in **CVPR 2022**

![](clims.png)

## 1. The quality of pseudo masks on PASCAL VOC2012.
| Method                | backbone | CAMs | + RW |
|:---------------------:|:---:|:----:|:----:|
| AdvCAM                | R50 | 55.6 | 68.0 |
| **CLIMS**                 | R50 | 56.6 | 70.5 |

## 2. Evaluation results on PASCAL VOC2012.
| Method                | Supervision| val | test |
|:---------------------:|:---:|:----:|:----:|
| AdvCAM                | S   | 68.1 | 68.0 |
| EDAM                  | I+S | 70.9 | 70.6 |
| **CLIMS**                 | S   | 69.3 | 68.7 | 
| **CLIMS**                 | S   | 70.4 | 70.0 |

(Initial CAMs, pseudo semantic masks, and pretrainde models can be download at [Google Drive](https://drive.google.com/drive/folders/1njCaolWacqSmw7HVNecwvCAMm7NsCFPq?usp=sharing))

If you are using our code, please consider citing our paper.

```
@article{clims,
  title={Cross Language Image Matching for Weakly Supervised Semantic Segmentation},
  author={Xie, Jinheng and Hou, Xianxu and Ye, Kai and Shen, Linlin},
  journal={arXiv preprint arXiv:2203.02668},
  year={2022}
}
```
