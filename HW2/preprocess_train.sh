# mkdir ./data
mkdir -p data

# train multiple choice
python preprocess_multiple_choice.py \
--data_dir $1 \
--output_dir ./data/multiple_choice_train.json

python preprocess_multiple_choice.py \
--data_dir $2 \
--output_dir ./data/multiple_choice_valid.json


# train QA
python preprocess_qa.py \
--data_dir $1 \
--context_dir $3 \
--output_dir ./data/QA_train.json

python preprocess_qa.py \
--data_dir $2 \
--context_dir $3 \
--output_dir ./data/QA_valid.json