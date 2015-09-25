rm -rf ./output
mkdir ./output

python -c 'import glm; glm.main_q4("tag.model", "tag_dev.dat", "./output/tag_dev.out")'

echo 'Output of Q4 has been saved at "./output/tag_dev.out"'
echo 'Evaluation report for Question 4 - Pretrained model with bigram and tag features'
python eval_tagger.py tag_dev.key ./output/tag_dev.out

python -c 'import glm; glm.main_q5("tag_train.dat", "suffix_tagger.model", "tag_dev.dat","./output/tag_dev_q5.out")'
echo 'Evaluation report for Question 5 - Model with bigram, tag, and suffix features'
python eval_tagger.py tag_dev.key ./output/tag_dev_q5.out

python -c 'import glm; glm.main_q6_1("tag_train.dat", "prefix_suffix_tagger.model", "tag_dev.dat","./output/tag_dev_q6_1.out", "1")'
echo 'Evaluation report for Question 6 - Model with bigram, tag, suffix and prefix features'
python eval_tagger.py tag_dev.key ./output/tag_dev_q6_1.out

python -c 'import glm; glm.main_q6_1("tag_train.dat", "prefix_suffix_tagger.model", "tag_dev.dat","./output/tag_dev_q6_2.out", "2")'
echo 'Evaluation report for Question 6 - Model with bigram, tag, suffix, prefix and number features'
python eval_tagger.py tag_dev.key ./output/tag_dev_q6_2.out

python -c 'import glm; glm.main_q6_1("tag_train.dat", "prefix_suffix_tagger.model", "tag_dev.dat","./output/tag_dev_q6_3.out", "3")'
echo 'Evaluation report for Question 6 - Model with bigram, tag, suffix, prefix and hyphen features'
python eval_tagger.py tag_dev.key ./output/tag_dev_q6_3.out