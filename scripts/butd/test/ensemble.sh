GPUID=$1
bs=32
backbone="butd"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD ensemble
bash run/gqa_ensemble_test.bash $GPUID $backbone/test/ensemble --load snap/gqa/pretrain/$backbone/mixup_best,snap/gqa/pretrain/$backbone/rp_best --test $subset --batchSize $bs --ensemble_method multiply --backbone $backbone

done
