# **Homework 2 ADL NTU 2022 Spring**

## **Reproduce prediction result**
### **`Method 1: directly use run.sh to predict testing data`**
```shell
bash download.sh
bash run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

example:
```shell
bash download.sh
bash run.sh ./data/context.json ./data/test.json ./data/test_submission.csv
```

<br>

### **`Method 2: step by step predict testing data`**
> ps: 後來才看到投影片裡面寫助教會直接執行 run.sh，一次完成前處理、Mutiple-Choice prediction、Question-Ansering prediction，如果想要從前處理一步一步來的話可以參考Method 2 （都寫了也捨不得刪掉@@）

### **Download Model and Config**
```shell
bash download.sh
```


### **Data Preprocess**
prepare proper data format for multiple-choice (store preprocessed data to ./data/multiple_choice_test.json)

```shell
bash preprocess_test.sh /path/to/test.json
```

example:
```shell
bash preprocess_test.sh ./data/test.json
```

### **Mutiple-Choice predict**
predict correct context and add to test data for question-answering (store prediction to ./data/QA_test.json)

```shell
bash test_multiple_choice.sh /path/to/context.json
```

example:
```shell
bash test_multiple_choice.sh ./data/context.json
```

### **Question-Ansering predict**
predict answer of test data 

```shell
bash test_qa /path/to/test_submission.csv
```

example:
```shell
bash test_qa ./data/test_submission.csv
```

<br>

## **Reproduce training process**

### **Data Preprocess**
prepare proper data format for multiple-choice and question-answering (store preprocessed data to ./data/multiple_choice_train.json, ./data/multiple_choice_valid.json, ./data/QA_train.json, ./data/QA_valid.json)

```shell
bash preprocess_train.sh /path/to/train.json /path/to/valid.json /path/to/context.json
```

example:
```shell
bash preprocess_train.sh ./data/train.json ./data/valid.json ./data/context.json
```

### **Mutiple-Choice train**

```shell
bash train_multiple_choice.sh /path/to/context.json
```

example:
```shell
bash train_multiple_choice.sh ./data/context.json
```

### **Question-Ansering train**

```shell
bash train_qa.sh
```