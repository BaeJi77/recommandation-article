# from https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
import textwrap as tr
from typing import List, Optional

import numpy as np
import openai
import pandas as pd
import plotly.express as px
from scipy import spatial


def read_openai_key(file_path):
    # 파일을 열고 내용을 읽은 후 출력합니다.
    try:
        with open(file_path, 'r') as file:
            api_key = file.read().strip()  # strip()을 사용하여 양쪽의 공백과 개행 문자를 제거합니다.

        return api_key
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


openai.api_key = read_openai_key("open_ai.key")


def get_embedding(text: str, model="text-similarity-davinci-001", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = openai.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def tsne_components_from_embeddings(
        embeddings: List[List[float]], n_components=2, **kwargs
) -> np.ndarray:
    """Returns t-SNE components of a list of embeddings."""
    # use better defaults if not specified
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"
    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)


def chart_from_components(
        components: np.ndarray,
        labels: Optional[List[str]] = None,
        strings: Optional[List[str]] = None,
        x_title="Component 0",
        y_title="Component 1",
        mark_size=5,
        **kwargs,
):
    """Return an interactive 2D chart of embedding components."""
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter(
        data,
        x=x_title,
        y=y_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)
