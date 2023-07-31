GPUID=$1
bs=32
backbone="lxmert"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD RP w/ hardUQ
bash run/gqa_conf_test.bash $GPUID $backbone/test/RP_with_hard_uq  --load snap/gqa/pretrain/$backbone/rp_harduq_best --test $subset --batchSize $bs 
done