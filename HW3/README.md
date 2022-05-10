# **Homework 3 ADL NTU 2022 Spring**

## **Reproduce prediction result**
```shell
bash download.sh
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
```

example:
```shell
bash download.sh
bash run.sh ./data/public.jsonl ./data/output_submission.jsonl
```


<br>


## **Reproduce training process**

### **Installation**
```shell
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```

```shell
bash train.sh /path/to/train_data.jsonl /path/to/validation_data.jsonl /path/to/output_dir
```
