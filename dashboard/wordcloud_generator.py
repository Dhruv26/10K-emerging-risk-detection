import base64
from collections import Counter
from io import BytesIO
from typing import Sequence

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def fig_to_uri(in_fig: plt.Figure, **save_args) -> str:
    out_img = BytesIO()
    in_fig.savefig(out_img, format="png", **save_args)
    encoded = base64.b64encode(out_img.getvalue()).decode('ascii')
    return "data:image/png;base64,{}".format(encoded)


def create_wordcloud(words: Sequence[str]):
    wc = WordCloud(
        width=2000,
        height=1200,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=1000,
        max_font_size=90,
        random_state=42,
        contour_width=1,
        contour_color="#119DFF",
    )
    wc.generate_from_frequencies(Counter(words))

    fig = plt.figure(figsize=[20, 12])
    ax = plt.imshow(wc.recolor(), interpolation="bilinear")
    plt.axis("off")
    return fig_to_uri(fig, bbox_inches="tight")
