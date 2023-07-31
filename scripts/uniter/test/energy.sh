GPUID=$1
bs=32
backbone="uniter"
for subset in "GQAUQ_testdev_questions_ClipEasy" #"GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do
# BUTD Energy
bash run/gqa_energy_test.bash $GPUID $backbone/test/energy --load snap/gqa/pretrain/$backbone/vanilla_best  --test $subset --batchSize $bs --backbone $backbone
done
