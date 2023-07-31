GPUID=$1
bs=32
backbone="butd"

for subset in "GQAUQ_testdev_questions_ClipEasy" "GQAUQ_testdev_questions_ClipHard" "GQAUQ_testdev_questions_PTEasy" "GQAUQ_testdev_questions_PTHard"
do

# BUTD FRCNN
bash run/gqa_fasterrcnn_test.bash $GPUID $backbone/test/frcnn --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --backbone $backbone

# BUTD MSP
bash run/gqa_conf_test.bash $GPUID $backbone/test/msp --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --backbone $backbone

# BUTD ODIN
bash run/gqa_odin_test.bash $GPUID $backbone/test/odin --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --temperature 1e5 --noise 1e-4 --backbone $backbone

# BUTD maha
#bash run/gqa_maha_test.bash $GPUID $backbone/maha --load snap/gqa/pretrain/$backbone/vanilla_best --test $subset --batchSize $bs --temperature 1e5 --noise 1e-4 --train train,valid --fast --backbone $backbone
bash run/gqa_maha_test.bash $GPUID butd_maha_test --load snap/gqa/pretrain/butd/vanilla_best --test $subset --batchSize $bs --temperature 1e5 --noise 1e-4 --train train,valid --fast --backbone butd

# BUTD Energy
bash run/gqa_energy_test.bash $GPUID $backbone/test/energy --load snap/gqa/pretrain/$backbone/vanilla_best  --test $subset --batchSize $bs --backbone $backbone

# BUTD qc
bash run/gqa_caption_test.bash $GPUID $backbone/test/qc --load snap/gqa/pretrain/$backbone/qc_best --load_gqa snap/gqa/$backbone/vanilla_best --test $subset --batchSize $bs --backbone $backbone

# BUTD resample
bash run/gqa_conf_test.bash $GPUID $backbone/test/resampling --load snap/gqa/pretrain/$backbone/resample_best --test $subset --batchSize $bs --backbone $backbone

# BUTD RP w/ hardUQ
bash run/gqa_conf_test.bash $GPUID $backbone/test/RP_with_hard_uq  --load snap/gqa/pretrain/$backbone/rp_harduq_best --test $subset --batchSize $bs --backbone $backbone

# BUTD RP
bash run/gqa_conf_test.bash $GPUID $backbone/test/RP  --load snap/gqa/pretrain/$backbone/rp_best --test $subset --batchSize $bs --backbone $backbone

# BUTD mixup
bash run/gqa_conf_test.bash $GPUID $backbone/test/mixup --load snap/gqa/pretrain/$backbone/mixup_best --test $subset --batchSize $bs --backbone $backbone

# BUTD ensemble
bash run/gqa_ensemble_test.bash $GPUID $backbone/test/ensemble --load snap/gqa/pretrain/$backbone/mixup_best,snap/gqa/pretrain/$backbone/rp_best --test $subset --batchSize $bs --ensemble_method multiply --backbone $backbone

done
