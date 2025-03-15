import pandas as pd

#Dicts from pxtextmining/pxtextmining/params.py
q_map = {
    #what_good:
    "What was good?": "what_good",

    #could_improve:
    "Is there anything we could have done better?": "could_improve",
    "How could we improve?": "could_improve",
    "What could we do better?": "could_improve",

    #nonspecific:
    "Please tell us why": "nonspecific",
    "Please tells us why you gave this answer?": "nonspecific",
    "FFT Why?": "nonspecific",
    "Please can you tell us why you gave your answer and what we could have done better?": "nonspecific",
    "Please describe any things about the 111 service that\r\nyou were particularly satisfied and/or dissatisfied with": "nonspecific",
    "Please describe any things about the 111 service that \nyou were particularly satisfied and/or dissatisfied with": "nonspecific",
    "Please describe any things about the 111 service that\nyou were particularly satisfied and/or dissatisfied with": "nonspecific",
    "Nonspecific": "nonspecific",
    "nonspecific": "nonspecific",
}
#Adapted to start at 0
sentiment_dict = {
    0: "very positive",
    1: "positive",
    2: "neutral",
    3: "negative",
    4: "very negative",
}


def clean_text(text: str) -> str:
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


def tweak_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tweak FFT spreadsheet, including question type categorisation.

    -Answer is lightly cleaned
    -Question type column added
    -Answer length columns added
    -Drop null answers
    -Return selected columns

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
        .assign(question_type=lambda df_: df_['FFT question'].map(q_map))

        #Clean answer. Light touch atm, might discard if using llm embeddings.
        .assign(answer_clean=lambda df_: df_['FFT answer'].fillna('').astype(str).map(clean_text))

        #Record length of the answer
        .assign(answer_word_len=lambda df_: df_['answer_clean'].str.split().str.len().astype(int))
        .assign(answer_char_len=lambda df_: df_['answer_clean'].str.len().astype(int))

        #Need text, so drop where answer_lengths==0
        .loc[lambda df_: df_['answer_char_len'].gt(0)]

        #Dates aren't formatted consistently. d/m sometimes swapped.
        # .assign(Date=lambda df_: pd.to_datetime(df_['Date'], format='%d/%m/%Y', errors='coerce'))

        #Select and order columns
        [['Comment ID', 'Trust', 'Respondent ID', 'Date',
          'question_type', 'answer_clean',
        #   'answer_char_len', 'answer_word_len',
        ]]
    )