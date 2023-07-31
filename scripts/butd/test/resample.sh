GPUID=$1
bs=32
backbone="butd"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD resample
bash run/gqa_conf_test.bash $GPUID $backbone/test/resampling --load snap/gqa/pretrain/$backbone/resample_best --test $subset --batchSize $bs --backbone $backbone
done
