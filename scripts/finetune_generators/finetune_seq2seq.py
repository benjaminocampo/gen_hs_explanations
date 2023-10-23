from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.preprocessing import flatten_dict
from datasets import Dataset, DatasetDict
from evaluate import load
import hydra
import pandas as pd
import torch
import logging
import mlflow
import shlex
import sys
import nltk
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def label_to_str(x):
    if x:
        return "hateful"
    else:
        return "non-hateful"


def process_data(data, all_targets):
    inputs = []
    outputs = []

    for _, row in data.iterrows():
        input_str = f"Given a message: {row['text']} 1) Label if it is hateful or non-hateful. 2) Label the target of hate. TARGETS = {', '.join(all_targets).lower()} 3) Generate an explanation of why the sentence is hateful or not. Output the answer in the following structure. Label: Target: Explanation: "
        output_str = f"Label: {label_to_str(row['label'])} Target: {row['sanitized_target'].lower()} Explanation: The message is {label_to_str(row['label'])} because it implies {row['implication']}."

        inputs.append(input_str)
        outputs.append(output_str)

    return inputs, outputs


def compute_metrics(eval_pred, tokenizer, metric):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions,
                                           skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip()))
        for label in decoded_labels
    ]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True,
                            use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        # Check if a GPU is available and if not, fall back to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load your dataset
        train_df = pd.read_csv(cfg.input.train_file)
        dev_df = pd.read_csv(cfg.input.dev_file)

        # TODO: this in the config file.
        TARGETS = [
            "BLACK PEOPLE", "WOMEN", "JEWS", "LGBTQ+", "MUSLIMS", "DISEASE",
            "ASIAN", "IMMIGRANTS", "WHITE PEOPLE", "AFRICAN"
        ]

        train_df = train_df[train_df["sanitized_label"].isin(TARGETS)
                            & ~train_df["implication"].isna()]
        train_df = train_df.rename(
            columns={"sanitized_label": "sanitized_target"})
        dev_df = dev_df[dev_df["sanitized_target"].isin(TARGETS)
                        & ~dev_df["implication"].isna()]

        train_inputs, train_outputs = process_data(train_df, TARGETS)
        dev_inputs, dev_outputs = process_data(dev_df, TARGETS)

        # Use the `load_dataset` function from Hugging Face to process these lists
        dataset = DatasetDict({
            'train':
            Dataset.from_dict({
                'input_text': train_inputs,
                'target_text': train_outputs
            }),
            'validation':
            Dataset.from_dict({
                'input_text': dev_inputs,
                'target_text': dev_outputs
            }),
        })

        model = T5ForConditionalGeneration.from_pretrained(
            cfg.input.pretrained_model_name_or_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(cfg.input.pretrained_model_name_or_path)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Tokenize our training dataset
        def tokenize_function(examples):
            model_inputs = tokenizer(examples['input_text'],
                                     padding='max_length',
                                     truncation=True,
                                     max_length=cfg.input.max_length)
            labels = tokenizer(examples['target_text'],
                               padding='max_length',
                               truncation=True,
                               max_length=cfg.input.max_label_length)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Tokenize each split of the dataset
        tokenized_train = dataset['train'].map(tokenize_function, batched=True)
        tokenized_validation = dataset['validation'].map(tokenize_function,
                                                         batched=True)

        # Load Rouge metric
        metric = load("rouge")

        # Define training arguments and train
        training_args = Seq2SeqTrainingArguments(
            f"{cfg.model.params.pretrained_model_name_or_path}_finetuned_on_sbic",
            per_device_train_batch_size=cfg.model.params.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.model.params.per_device_eval_batch_size,
            eval_accumulation_steps=cfg.model.params.eval_accumulation_steps,
            learning_rate=cfg.model.params.learning_rate,
            weight_decay=cfg.model.params.weight_decay,
            push_to_hub=cfg.model.params.push_to_hub,
            logging_dir=output_dir,
            logging_steps=cfg.model.params.logging_steps,
            eval_steps=cfg.model.params.eval_steps,
            save_steps=cfg.model.params.save_steps,
            save_total_limit=cfg.model.params.save_total_limit,
            num_train_epochs=cfg.model.params.num_train_epochs,
            predict_with_generate=cfg.model.params.predict_with_generate,
            evaluation_strategy=cfg.model.params.evaluation_strategy,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred, tokenizer, metric),
            data_collator=data_collator,
        )

        trainer.train()


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))

    if cfg.input.uri_path is not None:
        mlflow.set_tracking_uri(cfg.input.uri_path)
        assert cfg.input.uri_path == mlflow.get_tracking_uri()

    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    mlflow.set_experiment(cfg.input.experiment_name)
    mlflow.set_experiment_tag('mlflow.note.content',
                              cfg.input.experiment_description)

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Log params expects a flatten dictionary, since the configuration has nested
        # configurations (e.g. train.model), we need to use flatten_dict in order to
        # transform it into something that can be easilty logged by MLFlow.
        mlflow.log_params(
            flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run)


if __name__ == '__main__':
    main()
