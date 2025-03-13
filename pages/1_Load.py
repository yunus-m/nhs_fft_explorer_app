#%%

import streamlit as st
import pandas as pd
import numpy as np

from spreadsheet_data_handling import tweak_raw_df

st.title('Load and process FFT data')

uploaded_file = st.file_uploader('Load FFT data', type='csv')
run_proc_btn = None

if uploaded_file:
    #
    # Data has been loaded - show
    #
    df = pd.read_csv(uploaded_file, index_col=0)
    st.toast(f'Loaded ' + uploaded_file.name)

    st.write(f'**Original data** | {len(df)} rows, {df.shape[1]} columns. Showing first 100 rows.')
    st.dataframe(df.iloc[:100])

    #
    # Show tweaked data
    #
    df_tweaked = df.pipe(tweak_raw_df)
    st.write('**Formatted data**')
    st.dataframe(df_tweaked.iloc[:100])

    run_proc_btn = st.button('Process data')
else:
    st.warning('No data')

if run_proc_btn:
    st.spinner('Running', show_time=True)

    #
    #Load local model and tokenizer
    #
    import torch
    device = 'cpu'; torch.set_num_threads(4) #mimic CPU-only

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('../data/hfcheckpoint-tokenizer')
    model = AutoModelForSequenceClassification.from_pretrained('../data/hfcheckpoint-tinybert-sst5-fft')
    
    model.eval()
    model.to(device)

    #
    #FFT to Dataset.from_pandas
    #
    from model_utils import combine_question_answer
    from datasets import Dataset

    #Load from pandas
    fft_dataset = Dataset\
        .from_pandas(df_tweaked[['question_type', 'answer_clean']])\
        .remove_columns('__index_level_0__')
    
    #Merge q & a into single text sequence
    fft_dataset = fft_dataset\
        .map(combine_question_answer, batched=True)\
        .remove_columns(['question_type', 'answer_clean'])
    
    #Tokenize
    fft_tokenized = fft_dataset\
        .map(lambda batch: tokenizer(batch['q_and_a'], truncation=True, max_length=512), batched=True)\
        .remove_columns('q_and_a')
    
    #
    # Configure loader
    #
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from scipy.special import softmax

    dynamic_padding_collator = DataCollatorWithPadding(tokenizer)

    loader = DataLoader(
        fft_tokenized,  batch_size=32,
        pin_memory=False if device=='cpu' else True,
        shuffle=False, collate_fn=dynamic_padding_collator
    )

    #
    # Get embeddings
    # For efficiency, get embeddings & logits in a single pass aot separate passes
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

            #[B, L, emb] -> avg tokens' hidden rep -> [B, emb]
            # embeddings_mb = features.last_hidden_state.cpu().numpy().mean(axis=1)

            #[B, L, emb] -> final CLS rep -> [B, emb]
            embeddings_mb = features.pooler_output.cpu().numpy()

            #Pass remainder through for logits
            logits_mb = classifier_head(features.pooler_output)

            #Record results
            embeddings.append(embeddings_mb)
            logits.append(logits_mb)
    
    embeddings = np.concatenate(embeddings, axis=0)
    logits = np.concatenate(logits, axis=0)


    



