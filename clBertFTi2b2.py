# coding=utf-8
"""
Set up a Clinical BERT Feature Extraction & Finetuning pipeline
Application on 2010 i2b2 dataset.  The objective is to learn how to
fine-tune a pretrained model in preparation to use on the Moore
Project clinical narratives data.

Author: Sandeep Shetty 
Date: 22 Nov 2022
"""

from numpy.random import default_rng
from datasets import list_datasets, load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
import re
import torch
import collections
import transformers

# i2b2 Data
# -->>>>
dat = pd.read_csv("../conll_i2b2.csv", index_col=0, keep_default_na=False)
dat["Doc_Id"] = dat["Doc_Id"].str.replace("record-", "").astype(int)
dat = dat[~(dat["NER_Tag"] == "")]  # delete blanks
dat["ner_id"] = dat["NER_Tag"].map({"O": 0, "B-PROBLEM": 1, "I-PROBLEM": 2})


# Fixed Parameters
FRAC = 0.8
MAX_LENGTH = 20  # Model max length - need a way to parameterize this better


# Train-Validate Samples
# Results not replicable with no seed
# --->>>>

doc_names = dat["Doc_Id"].unique()
train_docs = np.random.choice(doc_names, int(FRAC * len(doc_names)))
validate_docs = list(set(doc_names) - set(train_docs))

train_sample = dat.loc[dat["Doc_Id"].apply(lambda x: x in train_docs)]
validate_sample = dat.loc[dat["Doc_Id"].apply(lambda x: x in validate_docs)]


# List of sentence and label
# Training Data
train_sentences = (
    train_sample.groupby(["Doc_Id", "Sent_id"])["Token"]
    .apply(lambda x: " ".join(x.tolist()).replace("\n", ""))
    .tolist()
)
train_ner_label = (
    train_sample.groupby(["Doc_Id", "Sent_id"])["ner_id"]
    .apply(lambda x: x.tolist())
    .tolist()
)
# Validation Data
validate_sentences = (
    validate_sample.groupby(["Doc_Id", "Sent_id"])["Token"]
    .apply(lambda x: " ".join(x.tolist()).replace("\n", ""))
    .tolist()
)
valid_ner_label = (
    validate_sample.groupby(["Doc_Id", "Sent_id"])["ner_id"]
    .apply(lambda x: x.tolist())
    .tolist()
)

# --->>> labels need to be fixed for dimensionality
# Change label length to match tokens (model length)
# Add an element in front for CLS

# Extend length of lists
# for items in train_ner_label:
# items.extend([-100] * (MAX_LENGTH - 1 - len(items)))
# items.insert(0, -100)

# for items in valid_ner_label:
# items.extend([-100] * (MAX_LENGTH - 1 - len(items)))
# items.insert(0, -100)


# Converting to convenient HuggingFace/PyTorch (?) dataset class
# ---->>>>
from datasets import Dataset, DatasetDict

train = {"text": train_sentences, "label": train_ner_label}
test = {"text": validate_sentences, "label": valid_ner_label}
ds_tr = Dataset.from_dict(train)
ds_test = Dataset.from_dict(test)
ner_data = DatasetDict({"train": ds_tr, "test": ds_test})


# Set up with BlueBERT MODEL for NER Classification
# 1. Conduct a feature extraction task
# Peng et al (2019) Transfer learning in Biomedical NLP...
# source: https://huggingface.co/bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16
# ---->>>>
model_ckpt = "distilbert-base-uncased"
model_blue = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
tokenize = AutoTokenizer.from_pretrained(
    model_blue, use_fast=True, add_prefix_space=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_old = AutoModel.from_pretrained(model_blue)
# assert isinstance(tokenize, transformers.PreTrainedTokenizerFast)

# Experiment with a single sentence
# ------>>>>>>>
# test_sent = train_sentences[500]

# tokenize - Providing an Id to each token from the Vocab
# encoded_text = tokenize(test_sent)
# print(encoded_text.input_ids)

# Get tokens back from the IDs
# tokens_txt = tokenize.convert_ids_to_tokens(encoded_text.input_ids)
# print(tokens_txt)
# ---->>>>>

# ---->>>>> Tokenize the Dataset
# Tokenizing requires adjustments to the labels in NER


def batch_tokenize(example):
    """To tokenize and align the labels by the same length as the
    tokenized sentence.  This function will apply to one example at a
    time.  # TODO: Convert this work on a batch.
    """

    tmp_sent = example["text"]
    sent_token = tmp_sent.split()
    # split pre-tokenizing helps with the token-to-word mapping; to
    # help resolve the label alignment problem

    # Consider dynamic padding -
    tokenized_text = tokenize(
        sent_token,  # example["text"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    word_ids = tokenized_text.word_ids()

    len_word_ids = len(word_ids)
    cur_label = example["label"]
    len_cur_label = len(cur_label)
     labels_new = []
    if len_cur_label == 1:  # Dates 2019-05-21 has NER tag ['O']
        for words in word_ids:
            if words is None:
                labels_new.append(-100)
            else:
                labels_new.append(cur_label[0])
    else:  # len greater than 1
        # for cases where there are all [O's]
        if len(set(cur_label)) == 1:
            for id in word_ids:
                if id is None:
                    labels_new.append(-100)
                else:
                    newlab = cur_label[0]
                    labels_new.append(newlab)
        else:
            for id in word_ids:
                if id is None:
                    labels_new.append(-100)
                else:
                    newlab = cur_label[id]
                    labels_new.append(newlab)
    tokenized_text["label"] = labels_new
    return tokenized_text


# --->>> Dynamically Pad the sentences using HF module

ner_encoded = ner_data.map(batch_tokenize)  # , batched=True)
ner_new = ner_encoded.rename_column("label", "label_ids")  # need label_ids

# add labels

# ner_data["train"].features["label_ids"].feature.names = ["O", "B-PROBLEM", "I-PROBLEM"]
# ner_data["test"].features["label_ids"].feature.names = ["O", "B-PROBLEM", "I-PROBLEM"]

# Data to Tensors
ner_new.set_format("torch", columns=["input_ids", "attention_mask", "label_ids"])


# 1. FEATURE EXTRACTION EXERCISE
# --->>>> Extract Hidden States

# 2. Define function to extract hidden states. Extract [CLS]
def extract_hidden_states(batch):
    # Place model inputs on the GPU (extracting only id, att, label)
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenize.model_input_names
    }  # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


# --->> Extracting features, a.k.a last hidden state
# 3. Apply above function to dataset - default batch_size=1000
ner_hidden = ner_encoded.map(extract_hidden_states, batched=True)

# --->>> Creating a feature matrix for classfication

X_train = np.array(ner_hidden["train"]["hidden_state"])
X_valid = np.array(ner_hidden["test"]["hidden_state"])
y_train = np.array(ner_hidden["train"]["label"])
y_valid = np.array(ner_hidden["test"]["label"], dtype=object)

print(X_train.shape, X_valid.shape)

# Note: Take this to a classification layer

# --->>>> FINE-TUNING TASK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# add classification layers on top of the last hidden states
# with softmax to convert to probabilities.. in the case of

# tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
# num_labels = 3
model_ckpt2 = "samrawal/bert-base-uncased_clinical-ner"
model = AutoModelForTokenClassification.from_pretrained(
    model_ckpt2, num_labels=3, ignore_mismatched_sizes=True
)

# Set up performance Metrics
from seqeval.metrics import f1_score
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.multioutput import MultiOutputClassifier

tags_dict = {"O": 0, "B-PROBLEM": 1, "I-PROBLEM": 2}
index2tag = {idx: tag for idx, tag in enumerate(tags_dict)}
tag2index = {tag: idx for idx, tag in enumerate(tags_dict)}

def align_predictions(predictions, label_ids):
    """From chapter 4 of NLP Transformers"""
    # # TODO: Fix this function
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
                labels_list.append(example_labels)
                preds_list.append(example_preds)
    return preds_list, labels_list


def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}



# Set up Trainer
# TODO: Consider unpacking the following piece.

from transformers import Trainer, TrainingArguments

batch_size = 64  # 128  # 64
logging_steps = len(ner_encoded["train"]) // batch_size
model_name = "model-blue-test-finetuned-NER"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ner_new["train"],
    eval_dataset=ner_new["test"],
    # tokenizer=tokenize,
)
trainer.train()
trainer.save_model()
results = trainer.evaluate()


# training  loop
"""
for batches, 
for epoch_index in range(args.num_epochs): train_state['epoch_index'] = epoch_index
# Iterate over training dataset
# setup: batch generator, set loss and acc to 0, set train mode on dataset.set_split('train')
batch_generator = generate_batches(dataset,
batch_size=args.batch_size, device=args.device)
running_loss = 0.0 running_acc = 0.0 classifier.train()
for batch_index, batch_dict in enumerate(batch_generator): # the training routine is 5 steps:
# step 1. zero the gradients optimizer.zero_grad()
# step 2. compute the output
y_pred = classifier(x_in=batch_dict['x_data'].float())
# step 3. compute the loss
loss = loss_func(y_pred, batch_dict['y_target'].float()) loss_batch = loss.item()
running_loss += (loss_batch ­ running_loss) / (batch_index + 1)
# step 4. use loss to produce gradients loss.backward()
# step 5. use optimizer to take gradient step optimizer.step()
"""
