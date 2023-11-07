from cgi import test
from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from src.preprocessing import flatten_dict
from tqdm import tqdm
from huggingface_hub import login
import hydra
import pandas as pd
import torch
import logging
import mlflow
import shlex
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )
        # Login to huggingface with read permissions.
        login(token=cfg.input.read_token)

        # Check if a GPU is available and if not, fall back to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load your dataset
        test_df = pd.read_csv(cfg.input.test_file)

        all_targets = test_df["target_ident"].str.upper().dropna().unique()
        labels = test_df["label_gold"].str.upper().unique()

        # Initialize the T5 model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(cfg.input.pretrained_model_name_or_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Replace the placeholders in the original prompt with the list contents:
        prompt = cfg.input.prompt.replace('TARGETS = ', f'TARGETS = {str(all_targets).lower()}')

        # Initialize an empty list to store explanations
        explanations = []

        # Batch size for checkpointing
        batch_size = cfg.input.checkpoint_batch_size

        # Loop through each message in the 'test_case' column
        for idx, hateful_message in tqdm(enumerate(test_df['text']), total=len(test_df)):

            # Combine the prompt and the hateful message
            input_text = prompt.replace("Given a message: ", f"Given a message: {hateful_message}")
            
            # Encode the text into tensor of integers using the appropriate tokenizer
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            
            # Generate output from the model
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=cfg.model.params.max_new_tokens,
                    num_beams=cfg.model.params.num_beams,
                    temperature=cfg.model.params.temperature,
                    do_sample=cfg.model.params.do_sample,
                    top_k=cfg.model.params.top_k,
                    no_repeat_ngram_size=cfg.model.params.no_repeat_ngram_size
                )
                
            # Decode and store the output text
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanations.append(decoded_output)
            
            # Checkpoint: Save intermediate results after each batch
            if (idx + 1) % batch_size == 0:
                test_df.loc[:idx, 'pred_exp'] = explanations[:idx + 1]
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
