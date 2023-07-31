# remove --tiny for full training
GPUID=$1
bash run/gqa_conf_finetune.bash $GPUID butd/train/vanilla --train train,valid --save_all --batchSize 64  --epochs 10 --backbone="butd" --tiny
