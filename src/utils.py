from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]
    
    true_predictions = [
        [label_names[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def tokenize_and_align_labels(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    original_labels = example["ner_tags"]
    word_ids = tokenized_inputs.word_ids()

    b_to_i_map = {}
    for i, label in enumerate(label_names):
        if label.startswith("B-"):
            i_label = "I-" + label[2:]
            if i_label in label_names:
                b_to_i_map[i] = label_names.index(i_label)

    new_labels = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != previous_word_id:
            new_labels.append(original_labels[word_id])
        else:
            original_label_id = original_labels[word_id]
            if original_label_id in b_to_i_map:
                new_labels.append(b_to_i_map[original_label_id])
            else:
                new_labels.append(original_label_id)
        
        previous_word_id = word_id

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs