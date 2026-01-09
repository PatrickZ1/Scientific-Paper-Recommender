from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from recomm_dataset import load_scidocs_cite, scidoc_cite_to_train_triplets
import pathlib

BASE_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
MODEL_OUT_DIR = pathlib.Path("./models/cross_encoder") / "tinybert_scidocs_cite"
CHECKPOINT_DIR = pathlib.Path("./.cross_enc_chkpt")

BATCH_SIZE = 128

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model = CrossEncoder(BASE_MODEL, device="cuda")

# TODO: use more than 1/100 of the training data for quick testing
train_dataset = scidoc_cite_to_train_triplets(
    load_scidocs_cite()["train"].shard(500, 0)
)

# TODO: Hard Example Mining or Loss Adjustment -> almost all negatives are too easy (will give a high score for paper from roughly the same field)

loss = losses.CachedMultipleNegativesRankingLoss(
    model, num_negatives=4, mini_batch_size=BATCH_SIZE
)
args = CrossEncoderTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    num_train_epochs=1,
    resume_from_checkpoint=True,
    per_device_train_batch_size=BATCH_SIZE,
)

trainer = CrossEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    args=args,
)

try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as e:
    if "checkpoint" in str(e):
        print("Restarting training from scratch (no valid checkpoint found).")
        trainer.train()
    else:
        raise e

# Save a clean, stable "final" model directory for evaluation loading
model.save(str(MODEL_OUT_DIR))
print(f"Saved fine-tuned CrossEncoder to: {MODEL_OUT_DIR.resolve()}")
