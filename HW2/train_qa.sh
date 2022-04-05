# train
python qa.py \
--do_train \
--do_eval \
--train_file ./data/QA_train.json \
--validation_file ./data/QA_valid.json \
--pad_to_max_length \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--output_dir ./qa \
# --max_train_samples 100 \
# --max_eval_samples 100
# debug

# --model_name_or_path bert-base-chinese \
