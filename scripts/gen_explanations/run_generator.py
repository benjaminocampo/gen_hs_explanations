import argparse
import subprocess
import itertools
import os
from inspect import cleandoc

# A lambda function to print strings with flushing, ensuring that they are immediately written to the output
printf = lambda s: print(s, flush=True)

# A counter to keep track of the number of commands that have resulted in errors
cmd_error_count = 0

def cmd(c):
    # Use the global cmd_error_count so that we can modify it inside this function
    global cmd_error_count
    
    # ANSI escape codes to set text color to red and to reset it back to default
    red_color = "\033[91m"
    end_color = "\033[0m"
    
    # Print the command that's being executed along with the current working directory
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    
    # Execute the command using os.system. If it returns a non-zero exit status (indicating an error),
    # print an error message and increase the error count.
    if os.system(c):  # Command returned != 0 (which means an error occurred)
        
        # Print a generic error message
        printf(
            f"{red_color}>>> [ERROR] there was an error in command:{end_color}"
        )
        
        # Print the specific command that caused the error along with the directory it was executed in
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        
        # Increment the error count
        cmd_error_count += 1
        
        # Exit the program
        exit()


def oar_submission_text_gpu(run_name, run_cmd):
    """
    Generates the OAR submission command string for GPU tasks.

    :param run_name: The name of the run.
    :param run_cmd: The command to run on the cluster.
    :return: The full OAR submission command string.
    """
    return cleandoc(f"""
        oarsub -p "gpu='YES' and gpucapability>='5.0' and gpumem>=15000" \
        -l /gpunum=1,nodes=1,walltime=1:00:00 \
        --stdout={run_name}.out \
        --stderr={run_name}.err \
        -q besteffort \
        'conda activate gen_hs_explanations ; {run_cmd}'
    """)

def main():
    # Create a parser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-on-cluster", action="store_true", 
                        help="Flag to indicate if the task should run on the cluster")
    args = parser.parse_args()

    # List of pretrained models and dataset names for the task
    pretrained_models = ['EleutherAI/gpt-neox-20b']
    dataset_names = ['hatecheck']

    # Loop through all combinations of pretrained models and dataset names
    for pretrained_model, dataset_name in itertools.product(pretrained_models, dataset_names):
        run_name = f'run_{pretrained_model}_{dataset_name}'
        run_name = run_name.replace("/", "_")
        run_cmd = f"python gen_explanations.py ++input.pretrained_model_name_or_path={pretrained_model} ++input.test_file=../../data/{dataset_name}_test.csv  input.run_name={run_name}"

        # If the run-on-cluster flag is set, submit the task to the OAR scheduler
        if args.run_on_cluster:
            cmd(oar_submission_text_gpu(run_name, run_cmd))
        else:
            # Otherwise, execute the command locally
            cmd(run_cmd)

if __name__ == '__main__':
    # Entry point for the script
    main()
