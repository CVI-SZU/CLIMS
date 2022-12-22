# Please specify config file and experiment name
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=7 python main_v2.py train --config-path configs/voc12_imagenet_pretrained.yaml
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=7 python main_v2.py test --config-path configs/voc12_imagenet_pretrained.yaml --model-path data/models/voc12_imagenet_pretrained/deeplabv2_resnet101_msc/train_aug/checkpoint_30000.pth
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=7 python main_v2.py crf --config-path configs/voc12_imagenet_pretrained.yaml
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=7 python eval.py --config_path configs/voc12_imagenet_pretrained.yaml --model_path data/models/voc12_imagenet_pretrained/deeplabv2_resnet101_msc/train_aug/checkpoint_30000.pth
# submit results.tar.gz to the evaluation server
cd data/features/voc12_imagenet_pretrained/deeplabv2_resnet101_msc/test/
tar -czf results.tar.gz results