mkdir -p ckpt/intent
mkdir -p ckpt/slot

wget https://www.dropbox.com/s/t9zdwtuccuvswnj/best-model.pth?dl=1 -O ckpt/intent/best-model.pth
wget https://www.dropbox.com/s/ay9qdwqzdkhtaat/best-model-79571.pth?dl=1 -O ckpt/slot/best-model.pth

mkdir -p cache/intent

wget https://www.dropbox.com/s/k585zndw60ek9ca/embeddings.pt?dl=1 -O cache/intent/embeddings.pt
wget https://www.dropbox.com/s/kwhsdczpsb9f4jz/intent2idx.json?dl=1 -O cache/intent/intent2idx.json
wget https://www.dropbox.com/s/ped8k4xyzkru615/vocab.pkl?dl=1 -O cache/intent/vocab.pkl

mkdir -p cache/slot
wget https://www.dropbox.com/s/97xw3cu3c6i2f9s/embeddings.pt?dl=1 -O cache/slot/embeddings.pt
wget https://www.dropbox.com/s/gdlcfd24uc1kjh4/tag2idx.json?dl=1 -O cache/slot/tag2idx.json
wget https://www.dropbox.com/s/y45dclyo40tyjre/vocab.pkl?dl=1 -O cache/slot/vocab.pkl

