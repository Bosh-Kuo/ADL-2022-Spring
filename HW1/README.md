# Homework 1 ADL NTU 2022 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Reproduce prediction result
### Prepare data
Make sure there is data folder in current directory

### Download my best model and cache
```shell 
bash downlod.sh
``` 

### **Intent detection**
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

### **Slot tagging**
```shell
bash ./slot_tagging.sh /path/to/test.json /path/to/pred.csv
```

---
## Reproduce training process
### Preprocessing
```shell
# To preprocess intent detectionay and slot tagging datasets
bash preprocess.sh
```
add "PAD":9 to the last line of cache/slot/tag2idx.json.
### Intent detection
```shell
python train_intent.py
```

---

### Slot tagging
```shell
python train_slot.py
```

