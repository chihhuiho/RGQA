GPUID=$1
bs=32
backbone="butd"
for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD ODIN
bash run/gqa_odin_test.bash $GPUID $backbone/test/odin --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --temperature 1e5 --noise 1e-4 --backbone $backbone
done
