import argparse
from calendar import different_locale
import json
from collections import defaultdict
import spacy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="../train_output.jsonl")
    parser.add_argument("--output_file", type=str, default="./data/train/transition_text.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    with open("../keywords.json") as f:
        keywords = json.load(f)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # lemmatize words in keywords
    for key, val in keywords.items():
        # separate words by its length (one, others) ex: one: rice, multi: ice cream
        one_lemma = []
        multi_lemma = []
        for word in val:
            split = [token.lemma_ for token in nlp(word)]
            if len(split) >= 2:
                multi_lemma.append(" ".join(split))
            else:
                one_lemma.append(split[0])
            keywords[key] = [one_lemma, multi_lemma]
    
    # chit-chat file
    with open(args.data_file, "r") as f:
        dialog = [json.loads(line) for line in f]
    
    data = []
    for d in tqdm(dialog):
        # start with the second utterance from the simulator
        for index in range(2, len(d["dialog"])):
            lemma_utterance_last = [token.lemma_ for token in nlp(d["dialog"][index-2])]
            lemma_utterance = [token.lemma_ for token in nlp(d["dialog"][index])]
            service_hits = defaultdict(int)
            intentSet_last = set()
            intentSet = set()
            for key, (one, multi) in keywords.items():
                # single-word keyword in utterance 
                intersection_last = set(one) & set(lemma_utterance_last)
                intersection = set(one) & set(lemma_utterance)
                
                # check whether the word, the length is bigger than 2, is in the utterance
                for m in multi:
                    unsplit_utterance_last = " ".join(lemma_utterance_last)
                    unsplit_utterance = " ".join(lemma_utterance)
                    if m in unsplit_utterance_last:
                        intersection_last.add(m)
                    if m in unsplit_utterance:
                        intersection.add(m)

                service_hits[key] += len(intersection)
                if len(intersection_last): 
                    intentSet_last.add(key)
                if len(intersection): 
                    intentSet.add(key)

            # Is there a keyword in this utterance
            topic_is_different = not (intentSet_last & intentSet)
            if topic_is_different and sum(service_hits.values()) != 0:
                """
                user: u1(topic1)
                system: s1(topic1)
                user: u2(topic1, have intent)
                system: s2(topic1, transition)
                user: u3(topic2)
                """
                intent = [key for key in intentSet]  # keywords in hit utterance(u1)
                target = d["dialog"][index-1]  # transition turn (s2)
                source = f'{d["dialog"][index-2]} <blank> {d["dialog"][index]}'  # (u2, u3)
                data.append({"source": source, "target":target, "intent":intent}) 

    json_data = {"data": data}
    json.dump(json_data, open(args.output_file, 'w'),indent=2)  
