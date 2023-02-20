OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=6 python main.py train --config-path configs/voc12_coco_pretrained.yaml
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=6 python main.py test --config-path configs/voc12_coco_pretrained.yaml --model-path data/models/voc12_coco_pretrained/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=6 python main.py crf --config-path configs/voc12_coco_pretrained.yaml
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=6 python eval.py --config_path configs/voc12_coco_pretrained.yaml --model_path data/models/voc12_coco_pretrained/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
# submit results.tar.gz to the evaluation server
cd data/features/voc12_coco_pretrained/deeplabv2_resnet101_msc/test/
tar -czf results.tar.gz results
