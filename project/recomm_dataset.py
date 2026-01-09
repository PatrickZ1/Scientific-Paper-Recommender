from datasets import load_dataset, Dataset
from ir_measures import Qrel
from langchain_core.documents import Document
from tqdm import tqdm


def load_scidocs_cite():
    """SciDocs - Citation Prediction (triplets: query/positive/negative)."""
    return load_dataset(
        "allenai/scirepeval",
        "cite_prediction_new",
    )


def scidoc_cite_to_train_triplets(dataset: Dataset) -> Dataset:
    """Convert SciDocs citation prediction dataset to triplet format."""

    def combine(text_fields):
        title = text_fields["title"] if text_fields["title"] is not None else ""
        abstract = (
            text_fields["abstract"] if text_fields["abstract"] is not None else ""
        )
        return f"{title} [SEP] {abstract}"

    def to_triplet(example):
        return {
            "query": combine(example["query"]),
            "pos": combine(example["pos"]),
            "neg": combine(example["neg"]),
        }

    return dataset.map(to_triplet, remove_columns=dataset.column_names)


def load_relish():
    """ReLiSH - Relevant Literature Search."""
    return load_dataset(
        "allenai/scirepeval",
        "relish",
    )


def scidoc_cite_to_q_doc_qrel(dataset: Dataset):
    """Convert SciDocs citation prediction dataset to lists of queries, docs, and qrels usable for evaluation with ir_measures."""
    queries, docs = {}, {}
    qrels = []

    for item in tqdm(dataset, desc="Processing dataset to qrel format"):
        query = item["query"]
        query_id = str(query["corpus_id"])
        query_text = f"{query['title']} [SEP] {query['abstract']}"
        queries[query_id] = query_text

        pos = item["pos"]
        pos_id = str(pos["corpus_id"])
        pos_text = f"{pos['title']} [SEP] {pos['abstract']}"
        if pos_id not in docs:
            docs[pos_id] = pos_text

        neg = item["neg"]
        neg_id = str(neg["corpus_id"])
        neg_text = f"{neg['title']} [SEP] {neg['abstract']}"
        if neg_id not in docs:
            docs[neg_id] = neg_text

        qrels.append(Qrel(query_id=query_id, doc_id=pos_id, relevance=1))
        qrels.append(Qrel(query_id=query_id, doc_id=neg_id, relevance=0))

    query_list = [Document(page_content=text, id=qid) for qid, text in queries.items()]
    doc_list = [Document(page_content=text, id=did) for did, text in docs.items()]
    return query_list, doc_list, qrels


def relish_to_q_doc_qrel(dataset: Dataset):
    """Convert ReLiSH dataset to lists of queries, docs, and qrels usable for evaluation with ir_measures."""
    queries, docs = {}, {}
    qrels = []

    for item in tqdm(dataset, desc="Processing dataset to qrel format"):
        query = item["query"]
        query_id = str(query["corpus_id"])
        query_text = f"{query['title']} [SEP] {query['abstract']}"
        queries[query_id] = query_text

        for doc in item["candidates"]:
            doc_id = str(doc["corpus_id"])
            doc_text = f"{doc['title']} [SEP] {doc['abstract']}"
            if doc_id not in docs:
                docs[doc_id] = doc_text

            # Use same notion of relevance as in the original paper (2 = relevant, 1 = partially relevant, 0 = not relevant; for metric calculation, we treat partially relevant as not relevant)
            relevance = 1 if doc["score"] == 2 else 0
            qrels.append(Qrel(query_id=query_id, doc_id=doc_id, relevance=relevance))

    query_list = [Document(page_content=text, id=qid) for qid, text in queries.items()]
    doc_list = [Document(page_content=text, id=did) for did, text in docs.items()]
    return query_list, doc_list, qrels
