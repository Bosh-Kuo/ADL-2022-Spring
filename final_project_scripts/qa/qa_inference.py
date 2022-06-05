import json
from argparse import ArgumentParser
from operator import itemgetter
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

from intent import intent_questions 

sgd_intents: Dict[str, str] = {
    f"{intent}-{q}": q
    for intent, questions in intent_questions.items()
    for q in questions
}


def classify_intent(example: Dict) -> Dict:

    instances = [
        (idx, intent, f"yes. no. {turn}", question)
        for idx, turn in enumerate(example)
        for intent, question in sgd_intents.items()
    ]
    results = nlp(
        question=list(map(itemgetter(-1), instances)),
        context=list(map(itemgetter(-2), instances)),
    )
    mappings = {i[:2]: r["answer"] for i, r in zip(instances, results)}
    new_dialog = [
        {
            "id": idx,
            "text": turn,
            "intent": list(
                set(
                    [
                        intent.split("-")[0]
                        for intent in sgd_intents
                        if mappings.get((idx, intent), None) == "yes."
                    ]
                )
            ),
        }
        for idx, turn in enumerate(example)
    ]

    return new_dialog


parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--data_file", type=str, default="qa_train_input.jsonl")
parser.add_argument("--output_file", type=str, default="qa_train_output.json")
args = parser.parse_args()

MODEL_NAME = "adamlin/distilbert-base-cased-sgd_qa-step5000"
REVISION = "negative_sample-questions"
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, revision=REVISION)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
nlp = QuestionAnsweringPipeline(model, tokenizer, device=args.device)

samples = [json.loads(i) for i in open(args.data_file, "r")]
utterances = []
for dialog in samples:  
    tempt = []
    for sentence in dialog["dialog"]:
        s = sentence["text"] 
        tempt.append(s)
    utterances.append(tempt)
intent_samples = []
for e in tqdm(utterances):
    intent_samples.append(classify_intent(e))

json.dump(intent_samples, open(args.output_file, "w"))
