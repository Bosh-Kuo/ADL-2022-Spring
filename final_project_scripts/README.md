# ADL_Final_Project
2022 ADL Final Project 

## **Basic Data**
Collect from simulator.py script provided by TA
- train_output.jsonl: 4819 dialogues
- validation_output.jsonl: 1009 dialogues
- test_output.jsonl: 980 dialogues

<br>

## **Hit Generator**
The `Hit Generator` try to mock user, which takes the past utterances as input and predict what user will say in user's next utterance in a different topic



### **Collect data**
Suppose the dialogue scenario between system and user like that:
```
user: u1(topic1)
                s1(topic1) :system 
user: u2(topic1, have intent)
                s2(topic1, transition) :system
user: u3(topic2)   
...
...   
```
- source : [u2] or [u1, s1, u2]
- target : [u3] 

``` shell
python make_HG_text.py --data_file ../train_output.jsonl --output_file ./data/train/hit_generator_text.json 
python make_HG_text.py --data_file ../validation_output.jsonl --output_file ./data/validation/hit_generator_text.json 
python make_HG_text.py --data_file ../test_output.jsonl --output_file ./data/test/hit_generator_text.json 
```

### **Train**
``` shell
python train_hit_generator.py --model_name_or_path t5-small --output_dir t5_hit_generator
```

### **Inference**
``` shell
python train_hit_generator.py --data_file ./data/train/hit_generator_text.json --output_file ./data/train/output.json
python train_hit_generator.py --data_file ./data/validation/hit_generator_text.json --output_file ./data/validation/output.json
python train_hit_generator.py --data_file ./data/test/hit_generator_text.json --output_file ./data/test/output.json
```
 
<br>

## **Transition**

### **Collect data**
- source : [u2, u3]
- target : [u2] 
  
```shell
python make_transition_text.py --data_file ../train_output.jsonl --output_file ./data/train/transition_text.json
python make_transition_text.py --data_file ../validation_output.jsonl --output_file ./data/validation/transition_text.json
python make_transition_text.py --data_file ../test_output.jsonl --output_file ./data/test/transition_text.json
```

### **Train**
``` shell
python train_transition_generator.py --model_name_or_path t5-small --output_dir t5_transition_generator
```

### **Inference**
``` shell
python train_transition_generator.py --data_file ./data/train/transition_text.json --output_file ./data/train/output.json
python train_transition_generator.py --data_file ./data/validation/transition_text.json --output_file ./data/validation/output.json
python train_transition_generator.py --data_file ./data/test/transition_text.json --output_file ./data/test/output.json
```

<br>

## **Run Simulator**
```shell
python simulator.py --num_chats 980 --split test --output t5_test_output.jsonl
```

### **Hit rate**
Hit rate:
- validation dataset: 0.874
- test dataset: 0.877


