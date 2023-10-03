import argparse
import subprocess
from inspect import cleandoc
import itertools

def oar_submission_text_gpu(run_name, run_cmd):
    return cleandoc(f"""
        oarsub -p "gpu='YES' and gpucapability>='5.0' and gpumem>=15000" -l /gpunum=1,nodes=1,walltime=9:00:00 --stdout={run_name}.out --stderr={run_name}.err -q default 'conda activate gen_hs_explanations ; {run_cmd}'
    """)

def main(run_on_cluster=False, pretrained_models=None, dataset_names=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_cluster", action="store_true")

    pretrained_models = pretrained_models or ['t5-small']
    dataset_names = dataset_names or ['hatecheck']

    for pretrained_model, dataset_name in itertools.product(pretrained_models, dataset_names):
        run_name = f'run_{pretrained_model}_{dataset_name}'
        run_cmd = f"python gen_explanations.py ++input.pretrained_model_name_or_path={pretrained_model} ++input.test_file=../../data/{dataset_name}_test.csv  input.run_name={run_name}"

        if run_on_cluster:
            cmd = oar_submission_text_gpu(run_name, run_cmd)
            subprocess.run(cmd, shell=True)
        else:
            subprocess.run(run_cmd, shell=True)

if __name__ == '__main__':
    main()
