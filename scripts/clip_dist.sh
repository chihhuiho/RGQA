for sub in GQAUQ_testdev_questions_ClipEasy GQAUQ_testdev_questions_ClipHard GQAUQ_testdev_questions_PTEasy GQAUQ_testdev_questions_PTHard
do
bash run/gqa_clip_test.bash $1 clip_dist --test $sub
done

