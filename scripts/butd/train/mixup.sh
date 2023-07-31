# remove --tiny for full training
GPUID=$1
beta=0.7
bash run/gqa_mixup_vis_finetune.bash $GPUID butd/train/mixup --train GQAUQ_train_questions_unsup,GQAUQ_valid_questions_unsup --batchSize 32 --save_all  --mixup_beta=$beta --backbone="butd"  --tiny
