# remove --tiny for full training
GPUID=$1
beta=3
bash run/gqa_mixup_vis_finetune.bash $GPUID uniter/train/mixup --train GQAUQ_train_questions_unsup,GQAUQ_valid_questions_unsup --batchSize 32 --save_all  --mixup_beta=$beta --backbone="uniter" --loadLXMERT snap/pretrained/uniter-base.pt --tiny
