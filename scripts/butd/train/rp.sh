# remove --tiny for full training
GPUID=$1
bash run/gqa_conf_finetune.bash $GPUID butd/train/RP --train GQAUQ_train_questions_unsup,GQAUQ_valid_questions_unsup --save_all --batchSize 64 --epochs 10  --backbone="butd"   --tiny
