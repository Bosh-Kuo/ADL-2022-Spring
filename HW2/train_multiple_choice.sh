# train
python multiple_choice.py \
--do_train \
--do_eval \
--train_file ./data/multiple_choice_train.json \
--validation_file ./data/multiple_choice_valid.json \
--context_file $1 \
--pad_to_max_length \
--model_name_or_path bert-base-chinese \
--output_dir ./multiple_choice \
# --debug

