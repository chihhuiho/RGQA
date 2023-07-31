GPUID=$1
bs=32
backbone="lxmert"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD FRCNN
bash run/gqa_fasterrcnn_test.bash $GPUID $backbone/test/frcnn --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs 
done
