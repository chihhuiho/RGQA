GPUID=$1
bs=32
backbone="lxmert"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD qc
bash run/gqa_caption_test.bash $GPUID $backbone/test/qc --load snap/gqa/pretrain/$backbone/qc_best --load_gqa snap/gqa/$backbone/vanilla_best --test $subset --batchSize $bs
done
