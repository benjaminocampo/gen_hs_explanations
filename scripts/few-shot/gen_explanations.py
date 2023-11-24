from pyexpat import model
from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from pathlib import Path
from src.preprocessing import flatten_dict
from tqdm import tqdm
from openai import OpenAI
import hydra
import pandas as pd
import logging
import mlflow
import shlex
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def label_to_str(x):
    if x:
        return "hateful"
    else:
        return "non-hateful"


def build_shot(row):
    shot = (
        f"Message: {row['text']}\n" +
        f"Label: {label_to_str(row['label'])}\n" +
        f"Target: {row['sanitized_target'].lower()}\n" +
        f"Explanation: The message is {label_to_str(row['label'])} because it implies {row['implication']}"
    )
    return shot


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        openai_client = OpenAI(api_key=cfg.input.openai_api_key)

        # Load your dataset
        train_df = pd.read_csv(cfg.input.train_file)
        train_df = train_df[~train_df["implication"].isna()]

        test_df = pd.read_csv(cfg.input.test_file)

        # Initialize an empty list to store explanations
        inputs = []
        explanations = []

        # Batch size for checkpointing
        batch_size = cfg.input.checkpoint_batch_size

        # Loop through each message in the 'test_case' column
        for idx, hateful_message in tqdm(enumerate(test_df['text']), total=len(test_df)):

            shots = train_df.sample(n=cfg.input.nof_shots, random_state=cfg.input.shots_random_state)
            shots = shots.apply(build_shot, axis=1).tolist()
            shots_text = "\n\n".join(shots)

            # Combine the prompt and the hateful message
            input_text = cfg.input.prompt_template.format(message=hateful_message, shots=shots_text)

            # Encode the text into tensor of integers using the appropriate tokenizer
            text = openai_client.completions.create(
                model=cfg.input.engine,
                prompt=input_text,
                max_tokens=cfg.input.max_token,
                n=cfg.input.n,
                stop=cfg.input.stop,
                temperature=cfg.input.temperature,
            ).choices[0].text
            # Generate output from the model

            inputs.append(input_text)
            explanations.append(text)

            # Checkpoint: Save intermediate results after each batch
            if (idx + 1) % batch_size == 0:
                test_df.loc[:idx, 'pred_exp'] = explanations[:idx + 1]
                test_df.loc[:idx, 'input_exp'] = inputs[:idx + 1]
                test_df.to_csv(output_dir / f'hatecheck_with_explanations_checkpoint_{idx + 1}.csv', index=False)
                print(f"Checkpoint saved for index {idx + 1}")

        # Add remaining explanations as a new column to the DataFrame
        test_df['pred_exp'] = explanations

        # Save the DataFrame to a new CSV file
        test_df.to_csv(output_dir / 'hatecheck_with_explanations_final.csv', index=False)

        # Log output directory with MLFlow
        mlflow.log_artifact(output_dir)


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
