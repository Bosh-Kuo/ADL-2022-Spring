# test
python qa.py \
--do_predict \
--test_file ./data/QA_test.json \
--output_csv $1 \
--pad_to_max_length \
--model_name_or_path ./qa \
--output_dir ./qa \
# --max_predict_samples 100
# debug
