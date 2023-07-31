GPUID=$1
bs=32
backbone="lxmert"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD maha
bash run/gqa_maha_test.bash $GPUID $backbone/test/maha --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --temperature 1e5 --noise 1e-4 --train train,valid --fast 
done
