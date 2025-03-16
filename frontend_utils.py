import numpy as np
import pandas as pd
from io import BytesIO

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, to_hex

from spreadsheet_data_handling import sentiment_dict

def cmap_to_colorscale(cmap : Colormap) -> list[str]:
    """
    Convert matplotlib colormap to a plotly colorscale (equally-spaced)

    Parameters
    ----------
    cmap : Colormap
        matplotlib colormap to convert
    
    Returns
    -------
    list of strings, where each string describes a colour's rgb composition
    """
    cmap_rgb = [cmap(i)[:3] for i in np.linspace(0., 1., cmap.N)]
    cmap_rgb_str = [
        'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * 255, g * 255, b * 255)
        for r, g, b in cmap_rgb
    ]
    return cmap_rgb_str


def discrete_plotly_colorscale(cmap_name : str, levels : int) -> list[tuple[float, str]]:
    """
    Get plotly-compatible discrete colorscale given matplotlib colormap's name

    Parameters
    ----------
    cmap_name : str
        Name of the matplotlib colormap
    
    levels : int
        Number of levels to discretise the colormap into

    Returns
    -------
    Plotly-compatible discrete colorscale
    """
    colorscale = cmap_to_colorscale(plt.get_cmap(cmap_name, levels))

    discrete_predictions_colorscale = [
        (0/5, colorscale[0]), (1/5, colorscale[0]),
        (1/5, colorscale[1]), (2/5, colorscale[1]),
        (2/5, colorscale[2]), (3/5, colorscale[2]),
        (3/5, colorscale[3]), (4/5, colorscale[3]),
        (4/5, colorscale[4]), (5/5, colorscale[4]),
    ]
    return discrete_predictions_colorscale


def style_sentiment_description(description : str, cmap_name='PiYG_r') -> str:
    """
    Pandas colour styler for standardised sentiment descriptions

    Parameters
    ----------
    description : str
        A description from `sentiment_dict` (e.g. "very positive")
    cmap_name : str
        Colour map to discretise and apply to the description cells
    
    Returns
    -------
    CSS-style string that maps the description to a colour map
    """
    cmap = plt.get_cmap(cmap_name, len(sentiment_dict))

    mapping = {
        description: to_hex(cmap( i/(cmap.N - 1) ))
        for i, description in enumerate(sentiment_dict.values())
    }
    return f'background-color: {mapping[description]}; color: {"white" if "very" in description else "black"}'
    # return f'background-color: {mapping[description]}; color: black'


def to_excel(df : pd.DataFrame) -> BytesIO:
    """
    Write a dataframe to Excel format, returning the results in a buffer

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to write to Excel format
    
    Returns
    -------
    A buffer containing the converted dataframe
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet_1')
    buffer.seek(0)

    return buffer