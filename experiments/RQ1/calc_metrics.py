import mlflow
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn.metrics import accuracy_score

# Name of the experiment to search for
experiment_name = "with_vs_without_explanations"

# Get the experiment by name
experiment = mlflow.get_experiment_by_name(experiment_name)

results = {}

if experiment:
    experiment_id = experiment.experiment_id
    # List all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    for _, run in runs.iterrows():
        run_id = run['run_id']
        # Access the run name from the tags (assuming 'mlflow.runName' is the correct tag key)
        run_name = run[
            'tags.mlflow.runName'] if 'tags.mlflow.runName' in run else None

        if not run_name:
            print(f"No runName tag found for run ID {run_id}")
            continue

        artifact_uri = run['artifact_uri']
        # Construct the path to preds.csv using runName
        preds_file_path = f"{artifact_uri}/{run_name}/preds.csv"

        # Replace "file://" from the path if it's there (common when using local storage)
        preds_file_path = preds_file_path.replace("file://", "")

        try:
            # Read the preds.csv file
            df = pd.read_csv(preds_file_path)

            # Extract label from prediction
            df["pred_label"] = (
                df["pred"]
                .apply(lambda p: p
                    .split("\n")[0]
                    .replace("Label:", "")
                    .strip(", .")
                    .lower())
            )

            df["pred_label_enc"] = df["pred_label"].replace({
                "hateful": 1,
                "non-hateful": 0
            })
            df["gold_label_enc"] = df["gold_label"].replace({
                "hateful": 1,
                "non-hateful": 0
            })
            accuracy = accuracy_score(df['pred_label_enc'],
                                      df['gold_label_enc'])
            results[run_name] = {}
            results[run_name]["overall"] = {}
            results[run_name]["overall"] = accuracy
            for func_name, group in df.groupby("functionality"):
                func_accuracy = accuracy_score(group['pred_label_enc'],
                                               group['gold_label_enc'])
                results[run_name][func_name] = {}
                results[run_name][func_name] = func_accuracy
        except Exception as e:
            print(
                f"Could not read preds.csv for run {run_name} (ID {run_id}). Error: {e}"
            )

    df_results = pd.DataFrame(results)
    df_results.columns = pd.MultiIndex.from_product([df_results.columns, ["accuracy"]])
    df_results = df_results.reset_index(names="functionality")

    mlflow.set_experiment("results_with_vs_without_explanations")
    mlflow.set_experiment_tag("mlflow.note.content", "RQ1 Results")

    with mlflow.start_run(run_name="ablation_all_runs") as run:
        with TemporaryDirectory() as tmpfile:
            output_dir = Path(tmpfile)    
            df_results.to_csv(output_dir / 'results_rq1.csv', index=False)            
            mlflow.log_artifact(output_dir)


else:
    print(f"No experiment found with name '{experiment_name}'")
