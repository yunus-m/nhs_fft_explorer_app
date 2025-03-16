import torch
import numpy as np

import pandas as pd
from scipy.special import softmax


@torch.no_grad()
def compute_metrics(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Compute the class predictions and loss.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch classification model (loaded on `device`)
    loader : torch.utils.data.DataLoader
        Data loader yielding batched samples
    device : str
        Device to map data to for computation

    Returns
    ----------
    tuple of predictions (np.ndarray), CE loss (float)
    """
    model.eval()

    val_preds = []
    cum_loss = 0

    for minibatch in loader:
        minibatch = {k: v.to(device) for k, v in minibatch.items()}
        outputs = model(**minibatch)

        preds = outputs.logits.argmax(dim=1).ravel().cpu()
        val_preds.extend(preds)
        cum_loss += outputs.loss.cpu().numpy() * len(outputs.logits)
    
    return np.array(val_preds), cum_loss / len(loader.dataset)


def logits_to_df(logits: np.ndarray, id2label: dict[int, str]) -> pd.DataFrame:
    """Format logits from a PyTorch model into a DataFrame, including entropy.

    Parameters
    ----------

    logits : np.ndarray (B, n_classes)
        Model's output logits
    id2label : dict[int, str]
       Mapping of class index to class name

    Returns
    ----------
    DataFrame with a row per sample recording probabilities, entropy, and prediction
    """
    label2id = {v: k for k, v in id2label.items()}
    
    return (
        pd.DataFrame(softmax(logits, axis=1), columns=id2label.values(), index=range(len(logits)))
        .assign(entropy=lambda df_: df_.apply(lambda row_: -np.sum(row_ * np.log2(row_)), axis=1))
        .assign(predicted_label=lambda df_: df_.iloc[:, 0:5].idxmax(axis=1))
        .assign(predicted_id=lambda df_: df_['predicted_label'].map(label2id))
    )


def combine_question_answer(batch) -> dict[str, list]:
    """Combine FFT question and answer into a single sequence
    
    Parameters
    ----------
    batch : HuggingFace Dataset LazyBatch
        dict with a batch of entries for each key
    
    Returns
    ----------
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
