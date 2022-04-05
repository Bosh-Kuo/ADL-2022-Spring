# test
python multiple_choice.py \
--do_test \
--test_file ./data/multiple_choice_test.json \
--output_file ./data/QA_test.json \
--context_file $1 \
--pad_to_max_length \
--model_name_or_path ./multiple_choice
