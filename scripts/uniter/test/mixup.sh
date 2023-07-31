GPUID=$1
bs=32
backbone="uniter"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do

# BUTD mixup
bash run/gqa_conf_test.bash $GPUID $backbone/test/mixup --load snap/gqa/pretrain/$backbone/mixup_best --test $subset --batchSize $bs --backbone $backbone

done
