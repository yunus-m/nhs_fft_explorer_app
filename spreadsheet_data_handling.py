import pandas as pd
import numpy as np

def clean_text(text):
    """Lightly process text to handle redundant codes.
    
    Parameters
    ----------
    text : str
        Passage of text to process
    
    Returns
    ----------
        Formatted text with some special punctuation mapped to standard equivalent.
    """
    replaced = (
        text.strip()
        .replace('\x0b', ' ')
        .replace('\n', ' ')
        .replace('¬', ' ')
        .replace('Â', '')
        .replace('Ã', '')
        .replace('â', '')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("‚", ",")
        .replace('“', '"')
        .replace('”', '"')
        .replace('„', '"')
        .replace('…', '...')
        .replace("™", 'tm')
        .replace('|🏻', '')
        # .lower() #Option in CountVectorizer. HF tokenizer will handle as relevant.
    )
    return ' '.join(replaced.split())


def tweak_generic(df):
    """Tweak FFT spreadsheet, including question type categorisation.

    -Answer is lightly cleaned
    -Question type column added
    -Answer length columns added
    -Drop null answers

    NB. Dates aren't consistently recorded. Sometimes d/m is swapped.

    Parameters
    ----------
    df : DataFrame
        df to format
    
    Returns
    ----------
    Tweaked DataFrame
    """
    return (
        df

        #FFT question -> {nonspecific, could_improve, what_good}
        .assign(question_type=lambda df_: df_['FFT question'].map(params.q_map))

        #Clean answer. Light touch atm, might discard if using llm embeddings.
        .assign(answer_clean=lambda df_: df_['FFT answer'].fillna('').astype(str).map(clean_text))

        #Record length of the answer
        .assign(answer_word_len=lambda df_: df_['answer_clean'].str.split().str.len().astype(int))
        .assign(answer_char_len=lambda df_: df_['answer_clean'].str.len().astype(int))

        #Need text, so drop where answer_lengths==0
        .loc[lambda df_: df_['answer_char_len'].gt(0)]

        #Dates aren't formatted consistently. d/m sometimes swapped.
        # .assign(Date=lambda df_: pd.to_datetime(df_['Date'], format='%d/%m/%Y', errors='coerce'))
    )


def tweak_for_sentiment(df):
    """Tweak FFT data for sentiment analysis
    
    -Generic tweak
    -Drop null sentiment entries
    -Add sentiment score description
    -Convert sentiment score to int
    -Drop columns not used for sentiment analysis

    Parameters
    ----------
    df : DataFrame
        df to tweak for sentiment analysis
    
    Returns
    ----------
    Tweaked df filtered to sentiment-relevant records
    """
    return (
        df
        .pipe(tweak_generic)

        #Drop rows where `Comment sentiment` is NaN
        .loc[lambda df_: df_['Comment sentiment'].notna()]

        #Sentiment textual description from params.sentiment_dict
        .assign(sentiment_desc=lambda df_: df_['Comment sentiment'].map(params.sentiment_dict))

        .astype({'Comment sentiment': int})

        #Select and order columns
        [['Comment ID', 'Trust', 'Respondent ID', 'Date',
          'question_type', 'answer_clean',
          'answer_char_len', 'answer_word_len',
          'Comment sentiment', 'sentiment_desc',
        ]]
    )