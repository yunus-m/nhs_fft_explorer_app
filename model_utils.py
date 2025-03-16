import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
#Fix for docker deployment with streamlit
# https://github.com/VikParuchuri/marker/issues/442
# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
torch.classes.__path__ = []

from torch.utils.data import DataLoader

from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding
)
from datasets import Dataset


#Check if running as a bundled executable or not
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    #Running as PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    #Running from source
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

#Enforce CPU-only
device = 'cpu'
torch.set_num_threads(torch.get_num_threads()) #4 for typical laptop


def combine_question_answer(batch) -> dict[str, list]:
    """
    Combine FFT question and answer into a single sequence
    
    Parameters
    ----------
    batch : HuggingFace Dataset LazyBatch
        dict with a batch of entries for each key
    
    Returns
    -------
    dict with key "q_and_a" that lists the combined sequence per sample
    """
    question_map = {
        'what_good': 'Question: What was good?',
        'could_improve': 'Question: What could be improved?',
        'nonspecific': 'Feedback:'
    }

    questions = [question_map[q] for q in batch['question_type']]

    answers = [
        ('Answer: ' if question != 'nonspecific' else '') + answer
        for question, answer in
        zip(batch['question_type'], batch['answer_clean'])
    ]

    combined = [q + ' ' + a for q, a in zip(questions, answers)]

    return {'q_and_a': combined}


def forward_pass_from_checkpoint(
          df_tweaked : pd.DataFrame, model_dir : str, tokenizer_dir : str
    ) -> dict[str, np.ndarray]:
        """
        Forward pass tweaked data through a HF checkpointed model.

        Parameters
        ----------
        df_tweaked : pd.DataFrame
            Cleansed data, comprising question and answer columns.

        model_dir : str
            Path to the checkpointed model
        
        tokenizer_dir : str
            Path to the saved tokenizer

        Returns
        -------
        A dict of results with predictions, class probabilities, embeddings, and entropy.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
             os.path.join(bundle_dir, model_dir)
        )

        tokenizer = AutoTokenizer.from_pretrained(
             os.path.join(bundle_dir, tokenizer_dir)
        )
        
        model.to(device)
        model.eval()

        #
        #FFT to Dataset.from_pandas
        #
        #Load from pandas
        fft_dataset = Dataset\
            .from_pandas(df_tweaked)\
            .remove_columns('__index_level_0__')
        
        #Merge fft question and answer into single text sequence
        fft_dataset = fft_dataset\
            .map(combine_question_answer, batched=True)\
            .remove_columns(['Comment ID', 'question_type', 'answer_clean'])
        
        #Tokenize
        fft_tokenized = fft_dataset\
            .map(lambda batch: tokenizer(batch['q_and_a'], truncation=True, max_length=512), batched=True)\
            .remove_columns('q_and_a')
        
        #
        # Configure loader
        #
        dynamic_padding_collator = DataCollatorWithPadding(tokenizer)

        loader = DataLoader(
            fft_tokenized,  batch_size=32,
            pin_memory=False if device=='cpu' else True,
            shuffle=False, collate_fn=dynamic_padding_collator
        )

        #
        # Get embeddings
        # For efficiency, get embeddings & logits in a single pass aot two passes
        #
        embeddings = []
        logits = []

        classifier_head = torch.nn.Sequential(model.dropout, model.classifier)
        with torch.no_grad():
            for minibatch in loader:
                minibatch = {k: v.to(device) for k, v in minibatch.items()}

                features = model.bert(
                    input_ids=minibatch['input_ids'],
                    attention_mask=minibatch['attention_mask']
                )

                #[B, L, emb] -> final CLS rep -> [B, emb]
                embeddings_mb = features.pooler_output.cpu().numpy()

                #Pass remainder through for logits
                logits_mb = classifier_head(features.pooler_output)

                #Record results
                embeddings.append(embeddings_mb)
                logits.append(logits_mb)
        
        embeddings = np.concatenate(embeddings, axis=0)
        logits = np.concatenate(logits, axis=0)

        #
        # From logits, get: predictions, entropies, class probabilities
        #
        probabilities = softmax(logits, axis=1)
        
        predictions = logits.argmax(axis=1)
        class_probs = probabilities[np.arange(len(logits)), predictions]
        entropies = -(probabilities * np.log2(probabilities)).sum(axis=1)

        model_results = {
             'predictions': predictions,
             'class_probs': class_probs,
             'entropies': entropies,
             'embeddings': embeddings
        }
        return model_results


def logits_to_df(logits: np.ndarray, id2label: dict[int, str]) -> pd.DataFrame:
    """
    Format logits from a PyTorch model into a DataFrame, including entropy.

    Parameters
    ----------

    logits : np.ndarray (B, n_classes)
        Model's output logits
    id2label : dict[int, str]
       Mapping of class index to class name

    Returns
    -------
    DataFrame with a row per sample recording probabilities, entropy, and prediction
    """
    label2id = {v: k for k, v in id2label.items()}
    
    return (
        pd.DataFrame(softmax(logits, axis=1), columns=id2label.values(), index=range(len(logits)))
        .assign(entropy=lambda df_: df_.apply(lambda row_: -np.sum(row_ * np.log2(row_)), axis=1))
        .assign(predicted_label=lambda df_: df_.iloc[:, 0:5].idxmax(axis=1))
        .assign(predicted_id=lambda df_: df_['predicted_label'].map(label2id))
    )