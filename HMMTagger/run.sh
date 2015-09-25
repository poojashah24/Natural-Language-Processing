rm -rf ./output
mkdir ./output

python -c 'import hmm; hmm.run_program("ner_train_rare.dat", False, "ner_rare.counts","./output/prediction_file_simple","./output/trigram_probability_rare","./output/prediction_file_viterbi")'

echo 'Simple named entity tagger ouput has been saved in "./output/prediction_file_simple"'
echo 'Trigram probabilities have been saved in file "./output/trigram_probability_rare"'
echo 'HMM tagger ouput has been saved in "./output/prediction_file_viterbi"'

echo 'Evaluation report for Question 4 - Simple named entity tagger using _RARE_ word probabilities'
python eval_ne_tagger.py ner_dev.key ./output/prediction_file_simple

echo 'Evaluation report for Question 5 - HMM tagger using the Viterbi algorithm'
python eval_ne_tagger.py ner_dev.key ./output/prediction_file_viterbi

python -c 'import hmm; hmm.run_program("ner_train_rare_groups.dat", True,"ner_rare_groups.counts","./output/prediction_file_simple_groups","./output/trigram_probability_rare_groups","./output/prediction_file_viterbi_groups")'

echo 'For Question 6:'
echo 'Simple named entity tagger output has been saved in "./output/prediction_file_simple_groups"'
echo 'Trigram probabilities have been saved in file "./output/trigram_probability_rare_groups"'
echo 'HMM tagger ouput has been saved in "./output/prediction_file_viterbi_groups"'

echo 'Evaluation report for Question 6 - HMM tagger using the Viterbi algorithm with groups'
python eval_ne_tagger.py ner_dev.key ./output/prediction_file_viterbi_groups