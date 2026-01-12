import pathlib

from recomm_dataset import *
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

BASE_MODEL = "sentence-transformers/stsb-roberta-base-v2"
MODEL_OUT_DIR = pathlib.Path("./models/embedding") / "roberta_scidocs_cite"
CHECKPOINT_DIR = pathlib.Path("./.embedding_chkpt")

NUM_EVAL_PAIRS = 10000
BATCH_SIZE = 40
SAVE_EVERY_N_SAMPLES = 40_000
EVAL_EVERY_N_SAMPLES = 5_000
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = SentenceTransformer(BASE_MODEL, device="cuda")

    scidocs_cite = load_scidocs_cite()
    train_dataset = scidoc_cite_to_train_triplets(
        filter_scidocs_cite(scidocs_cite["train"], load_relish()["evaluation"])
    )

    examples = []
    for item in tqdm(
        scidocs_cite["validation"].shuffle(seed=42).select(range(NUM_EVAL_PAIRS // 2)),
        desc="Preparing evaluation examples",
    ):
        query = item["query"]["title"] + " " + (item["query"]["abstract"] or "")
        pos = item["pos"]["title"] + " " + (item["pos"]["abstract"] or "")
        neg = item["neg"]["title"] + " " + (item["neg"]["abstract"] or "")

        examples.append(InputExample(texts=[query, pos], label=1.0))
        examples.append(InputExample(texts=[query, neg], label=0.0))

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples,
        name="scidocs_cite_embedding_eval",
    )

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        num_train_epochs=1,
        resume_from_checkpoint=True,
        per_device_train_batch_size=BATCH_SIZE,
        dataloader_num_workers=4,  # change to 0 on AMD GPU to avoid HIP + multiprocessing + tensor sharing issues
        save_steps=int(SAVE_EVERY_N_SAMPLES / BATCH_SIZE),
        report_to=["tensorboard"],
        eval_strategy="steps",
        warmup_steps=500,
        weight_decay=0.01,
        bf16=True,
        eval_steps=int(EVAL_EVERY_N_SAMPLES / BATCH_SIZE),
        logging_steps=int(EVAL_EVERY_N_SAMPLES / (2 * BATCH_SIZE)),
        logging_first_step=True,
    )

    trainer = SentenceTransformerTrainer(
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

    model.save(str(MODEL_OUT_DIR))
    print(f"Saved fine-tuned Embedding Model to: {MODEL_OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
