rm -rf ./output
mkdir ./output

python -c 'import cky; cky.run_program("parse_train.dat", "parse_train_rare.dat", "cfg_rare.counts","./output/prediction_file","parse_dev.dat")'

echo 'Output of CKY algorithm has been saved at "./output/prediction_file"'
echo 'Evaluation report for Question 5 - CKY algorithm'
python eval_parser.py parse_dev.key ./output/prediction_file

python -c 'import cky; cky.run_program("parse_train_vert.dat", "parse_train_vert_rare.dat", "cfg_vert_rare.counts","./output/prediction_file_vert","parse_dev.dat")'
echo 'Evaluation report for Question 6 - CKY algorithm using vertical markovization'
echo 'Output of CKY algorithm has been saved at "./output/prediction_file_vert"'
python eval_parser.py parse_dev.key ./output/prediction_file_vert
