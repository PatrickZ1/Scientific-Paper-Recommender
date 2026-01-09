from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
    evaluation,
)
from sentence_transformers.training_args import BatchSamplers

from recomm_dataset import *
import pathlib

BASE_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
MODEL_OUT_DIR = pathlib.Path("./models/cross_encoder") / "tinybert_scidocs_cite"
CHECKPOINT_DIR = pathlib.Path("./.cross_enc_chkpt")

# Number of pairs from the evaluation set to use for evaluation during training
NUM_EVAL_PAIRS = 10000

# NOTE: Adjust if using a GPU with more/less memory
BATCH_SIZE = 40

SAVE_EVERY_N_SAMPLES = 40_000  # Save a checkpoint every N training samples (will be converted to steps based on batch size)
EVAL_EVERY_N_SAMPLES = 5_000  # Evaluate every N training samples (will be converted to steps based on batch size); training metrics will be logged twice as often

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model = CrossEncoder(BASE_MODEL, device="cuda")

scidocs_cite = load_scidocs_cite()

# TODO: use more than 1/100 of the training data for quick testing
train_dataset = scidoc_cite_to_train_triplets(
    filter_scidocs_cite(scidocs_cite["train"], load_relish()["evaluation"])
)

# NOTE: Due to computational constraints, we use only a subset of the evaluation set for evaluation during training (5000 pairs take around 2sec on my machine)
eval_pairs = []
eval_labels = []
for item in tqdm(
    scidocs_cite["validation"].shuffle(seed=42).select(range(NUM_EVAL_PAIRS // 2)),
    desc="Preparing evaluation pairs",
):
    query = item["query"]["title"] + " " + (item["query"]["abstract"] or "")
    pos = item["pos"]["title"] + " " + (item["pos"]["abstract"] or "")
    neg = item["neg"]["title"] + " " + (item["neg"]["abstract"] or "")
    eval_pairs.append([query, pos])
    eval_labels.append(1)
    eval_pairs.append([query, neg])
    eval_labels.append(0)

evaluator = evaluation.CrossEncoderClassificationEvaluator(
    eval_pairs,
    eval_labels,
    name="scidocs_cite_eval",
)

# NOTE: Cached Version is slower to train and the batch size is sufficient to not benefit from reducing GPU memory usage
loss = losses.MultipleNegativesRankingLoss(model)
# loss = losses.CachedMultipleNegativesRankingLoss(model)

args = CrossEncoderTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    num_train_epochs=1,
    resume_from_checkpoint=True,
    per_device_train_batch_size=BATCH_SIZE,
    dataloader_num_workers=4,
    save_steps=int(SAVE_EVERY_N_SAMPLES / BATCH_SIZE),
    report_to=["tensorboard"],
    eval_strategy="steps",
    warmup_steps=500,  # A linear warmup of the learning rate
    weight_decay=0.01,  # Change from default (0.0) to add a bit of weight decay
    bf16=True,  # Use mixed precision training
    eval_steps=int(EVAL_EVERY_N_SAMPLES / BATCH_SIZE),  # Evaluate every N steps
    logging_steps=int(
        EVAL_EVERY_N_SAMPLES / (2 * BATCH_SIZE)
    ),  # Log twice as often as evaluation
    logging_first_step=True,
)

trainer = CrossEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
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
