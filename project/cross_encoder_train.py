from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

from recomm_dataset import load_scidocs_cite, scidoc_cite_to_triplets

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", max_length=512)

# TODO: use more than 1/100 of the training data for quick testing
train_dataset = scidoc_cite_to_triplets(load_scidocs_cite()["train"].shard(100, 0))

loss = losses.CachedMultipleNegativesRankingLoss(model)
args = CrossEncoderTrainingArguments(
    output_dir="./cross-encoder-checkpoints",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    num_train_epochs=1,
    resume_from_checkpoint=False,
)
trainer = CrossEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    args=args,
)
trainer.train()

query = "What is the capital of France?"
passages = [
    "Berlin is the capital of Germany.",
    "Berlin is the capital of Germany.",
    "The capital of France is Paris.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "It is a sunny day.",
    "It is France's capital city.",
    "It is Paris.",
    "It is the city of love.",
]

ranks = model.rank(query, passages)

print("Query:", query)
for rank in ranks:
    print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")
