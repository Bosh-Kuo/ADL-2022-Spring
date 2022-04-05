# mkdir ./data
mkdir -p data

# preprocess test.json  
# store new data for multiple-choice task to ./data
python preprocess_multiple_choice.py \
--data_dir $2 \
--output_dir ./data/multiple_choice_test.json

# multiple choice prediction 
# store new data for question-answering task to ./data
python multiple_choice.py \
--do_test \
--test_file ./data/multiple_choice_test.json \
--output_file ./data/QA_test.json \
--context_file $1 \
--pad_to_max_length \
--model_name_or_path ./multiple_choice

# question-answering prediction
# store QA output prediction to $3 (/path/to/pred/prediction.csv)
python qa.py \
--do_predict \
--test_file ./data/QA_test.json \
--output_csv $3 \
--pad_to_max_length \
--model_name_or_path ./qa \
--output_dir ./qa \




