import itertools
from src.run import cmd


def main():

    models = [
        #"gpt-3.5",
        #"gpt-4",
        "mistral",
        #"gpt-3.5-ft",
        #"mistral-ft",
        #"bart-ft",
        #"bart",
    ]

    prompts = [
        "with-exp",
        "without-exp",
        #"fs-with-exp",
        #"fs-without-exp"
    ]

    datasets = [
        "hatecheck",
        #"latenthatred"
    ]

    for model, prompt, dataset in itertools.product(models, prompts, datasets):
        run_name = f'run_{model}_{prompt}_{dataset}'
        run_cmd = f"python gen_preds.py input.run_name={run_name} model={model} prompt={prompt} dataset={dataset}"
        cmd(run_cmd)

if __name__ == '__main__':
    # Entry point for the script
    main()
