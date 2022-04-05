mkdir multiple_choice
mkdir qa

# multiple-choice
wget https://www.dropbox.com/s/yxxsaxlpju88p6b/config.json?dl=1 -O ./multiple_choice/config.json
wget https://www.dropbox.com/s/1afq55gzih53qfx/pytorch_model.bin?dl=1 -O ./multiple_choice/pytorch_model.bin
wget https://www.dropbox.com/s/ui9dcn3fc62q48o/special_tokens_map.json?dl=1 -O ./multiple_choice/special_tokens_map.json
wget https://www.dropbox.com/s/roouej6anp70174/tokenizer_config.json?dl=1 -O ./multiple_choice/tokenizer_config.json
wget https://www.dropbox.com/s/87flvqrnptleief/tokenizer.json?dl=1 -O ./multiple_choice/tokenizer.json
wget https://www.dropbox.com/s/bo5zx6a7lyt61cz/vocab.txt?dl=1 -O ./multiple_choice/vocab.txt

# question-answering
wget https://www.dropbox.com/s/6qqgw23rxzwf01n/config.json?dl=1 -O ./qa/config.json
wget https://www.dropbox.com/s/p1t3iiohm1wespy/pytorch_model.bin?dl=1 -O ./qa/pytorch_model.bin
wget https://www.dropbox.com/s/6iz6yrccoc3egrl/special_tokens_map.json?dl=1 -O ./qa/special_tokens_map.json
wget https://www.dropbox.com/s/budgysnc3qc303g/tokenizer_config.json?dl=1 -O ./qa/tokenizer_config.json
wget https://www.dropbox.com/s/ftfqfy0hp4mp5p3/tokenizer.json?dl=1 -O ./qa/tokenizer.json
wget https://www.dropbox.com/s/aigzzt1afot6sko/vocab.txt?dl=1 -O ./qa/vocab.txt