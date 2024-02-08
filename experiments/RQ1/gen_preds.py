from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from src.preprocessing import flatten_dict
from src.usd_models import GPTModel, Mistral

import hydra
import pandas as pd
import logging
import mlflow
import shlex
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

def build_shot(row, no_exp=True):
    shot = (
        f"Message: {row['text']}\n" +
        f"Label: {row['label']}\n" +
        f"Target: {row['target'].lower()}" +
        ("" if no_exp else "\n" + f"Explanation: {row['gold_exp']}")
    )
    return shot

def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)
        output_dir = output_dir / run._info.run_name
        output_dir.mkdir(exist_ok=True)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        # Load your dataset
        test_df = pd.read_csv(cfg.dataset.path)

        # Initialize an empty list to store model's inputs and outputs
        inputs = []
        preds = []

        # Batch size for checkpointing
        batch_size = cfg.input.checkpoint_batch_size
        model = cfg.model.module(**cfg.model.params)

        # test_df = test_df.iloc[500:].reset_index(drop=True)

        # Build input_texts considering if few-shot or zero-shot is used
        if "fs" in run._info.run_name:
            train_df = pd.read_csv(cfg.prompt.params.fs_data_path)
            input_texts = []
            for message in test_df["text"]:
                shots = train_df.sample(n=cfg.prompt.params.nof_shots)
                shots = shots.apply(lambda s: build_shot(
                    s, "without-exp" in run._info.run_name),
                                    axis=1).tolist()
                shots_text = "\n\n".join(shots)
                input_texts.append(
                    cfg.prompt.params.template.format(message=message,
                                                      shots=shots_text))
        else:
            input_texts = [
                cfg.prompt.params.template.format(message=message)
                for message in test_df["text"]
            ]

        try:
            # Loop through each message in the 'text' column
            for idx, input_text in tqdm(enumerate(input_texts), total=len(input_texts)):

                # Generate output from the model
                text = model.generate(input_text)

                inputs.append(input_text)
                preds.append(text)

                # Checkpoint: Save intermediate results after each batch
                if (idx + 1) % batch_size == 0:
                    test_df.loc[:idx, 'pred'] = preds[:idx + 1]
                    test_df.loc[:idx, 'input_text'] = inputs[:idx + 1]
                    test_df.to_csv(
                        output_dir /
                        f'preds_checkpoint_{idx + 1}.csv',
                        index=False)
                    print(f"Checkpoint saved for index {idx + 1}")

            # Add remaining explanations as a new column to the DataFrame
            test_df['pred'] = preds
            test_df['input_text'] = inputs

            # Save the DataFrame to a new CSV file
            test_df.to_csv(output_dir / 'preds.csv', index=False)

            # Log output directory with MLFlow
            mlflow.log_artifact(output_dir)
        except Exception as err:
            # In case of an error, save predictions
            test_df.to_csv(output_dir / 'preds_unfinished.csv', index=False)
            mlflow.log_artifact(output_dir)
            logger.warning(f"Unfinished generation: {err}")
            logger.warning("Saving all the obtained generations so far.")



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
