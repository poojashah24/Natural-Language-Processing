python -c 'import ibmmodels; ibmmodels.main("original.de", "scrambled.en", "unscrambled.en", "alignments_model1", "alignments_model2", "foreign_words_output")'

echo 'Output of Q6 translation has been saved at "./unscrambled.en"'
echo 'Evaluation report for Question 6 - Finding translations'
python eval_scramble.py unscrambled.en original.en