# mkdir ./data
mkdir -p data

# preprocess test.json and store new data for multiple-choice task to ./data
python preprocess_multiple_choice.py \
--data_dir $1 \
--output_dir ./data/multiple_choice_test.json



