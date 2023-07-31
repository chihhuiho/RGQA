# remove --tiny for full training
GPUID=$1
bash run/gqa_conf_finetune.bash $GPUID uniter/train/RP_with_hard_uq --train GQAUQ_train_questions_unsup_hard,GQAUQ_valid_questions_unsup_hard --save_all --batchSize 64  --epochs 10  --backbone="uniter" --loadLXMERT snap/pretrained/uniter-base.pt --tiny
