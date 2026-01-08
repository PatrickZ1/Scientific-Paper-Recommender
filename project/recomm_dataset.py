from typing import Optional
from datasets import DatasetDict, load_dataset, Dataset


def load_scidocs_cite(
    *, cache_dir: Optional[str] = None, streaming: bool = False
) -> DatasetDict:
    """SciDocs - Citation Prediction (triplets: query/positive/negative)."""
    return load_dataset(
        "allenai/scirepeval",
        "cite_prediction_new",
        cache_dir=cache_dir,
        streaming=streaming,
    )


def scidoc_cite_to_triplets(dataset: Dataset) -> Dataset:
    """Convert SciDocs citation prediction dataset to triplet format."""

    def combine(text_fields):
        title = text_fields["title"] if text_fields["title"] is not None else ""
        abstract = (
            text_fields["abstract"] if text_fields["abstract"] is not None else ""
        )
        return f'"{title}"\n{abstract}'

    def to_triplet(example):
        return {
            "query": combine(example["query"]),
            "pos": combine(example["pos"]),
            "neg": combine(example["neg"]),
        }

    return dataset.map(to_triplet, remove_columns=dataset.column_names)


def load_relish(
    *, cache_dir: Optional[str] = None, streaming: bool = False
) -> DatasetDict:
    """ReLiSH - Relevant Literature Search."""
    return load_dataset(
        "allenai/scirepeval",
        "relish",
        cache_dir=cache_dir,
        streaming=streaming,
    )


__all__ = ["load_scidocs_cite", "load_relish"]
