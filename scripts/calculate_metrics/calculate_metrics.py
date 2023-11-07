import argparse
import pandas as pd
import evaluate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(data, functionality=None):
    metrics = ["bertscore", "bleurt", "bleu"]
    params = [{"model_type": "distilbert-base-uncased"}, {}, {"max_order": 1, "smooth": True}]
    val_to_output = ["f1", "scores", "bleu"]
    results = {}
    for metric_name, ps, out in zip(metrics, params, val_to_output):
        metric = evaluate.load(metric_name)
        if functionality:
            grouped_data = data.groupby(functionality)
            scores_by_group = {}
            for name, group in grouped_data:
                scores = metric.compute(predictions=group.reset_index()['pred_exps'], references=group.reset_index()['gold_exps'], **ps)
                scores_by_group[name] = np.mean(scores[out])
            results[metric_name] = scores_by_group
        else:
            scores = metric.compute(predictions=data['pred_exps'], references=data['gold_exps'], **ps)
            results[metric_name] = np.mean(scores[out])
    return results

def get_classification_metrics(data, pred_column, gold_column):
    precision, recall, f1, _ = precision_recall_fscore_support(data[gold_column], data[pred_column], average='micro', zero_division=0)
    accuracy = accuracy_score(data[gold_column], data[pred_column])
    return precision, recall, f1, accuracy

def main(csv_file):

    func_num = {
        "counter_quote_nh": "F20",
        "counter_ref_nh": "F21",
        "derog_dehum_h": "F3",
        "derog_impl_h": "F4",
        "derog_neg_attrib_h": "F2",
        "derog_neg_emote_h": "F1",
        "ident_neutral_nh": "F18",
        "ident_pos_nh": "F19",
        "negate_neg_nh": "F15",
        "negate_pos_h": "F14",
        "phrase_opinion_h": "F17",
        "phrase_question_h": "F16",
        "profanity_h": "F10",
        "profanity_nh": "F11",
        "ref_subs_clause_h": "F12",
        "ref_subs_sent_h": "F13",
        "slur_h": "F7",
        "slur_homonym_nh": "F8",
        "slur_reclaimed_nh": "F9",
        "target_group_nh": "F24",
        "target_indiv_nh": "F23",
        "target_obj_nh": "F22",
        "threat_dir_h": "F5",
        "threat_norm_h": "F6"
    }
    data = pd.read_csv(csv_file)
    data["func_num"] = data["functionality"].replace(func_num)

    data = data[~data["gold_exps"].isna()]
    data["pred_labels"] = data["pred_labels"].replace({"hateful": 1, "non-hateful": 0})
    data["gold_labels"] = data["gold_labels"].replace({"hateful": 1, "non-hateful": 0})

    # DataFrame for overall results
    overall_results = {}

    # Evaluate generations overall and store results
    overall_results = calculate_metrics(data)

    # Classification metrics overall and store results
    precision, recall, f1, accuracy = get_classification_metrics(data, 'pred_labels', 'gold_labels')
    overall_results['precision'] = precision
    overall_results['recall'] = recall
    overall_results['f1'] = f1
    overall_results['accuracy'] = accuracy

    overall_df = pd.DataFrame([overall_results])
    print("Overall Results:")
    print(overall_df)

    # DataFrame for results by functionality
    functionality_results = {}

    # Evaluate generations by functionality and store results
    gen_scores_functionality = calculate_metrics(data, functionality='functionality')
    for metric, scores_by_func in gen_scores_functionality.items():
        for func, score in scores_by_func.items():
            functionality_results.setdefault(func, {})[metric] = score

    # Classification metrics by functionality and store results
    grouped_data = data.groupby('functionality')
    for name, group in grouped_data:
        precision, recall, f1, accuracy = get_classification_metrics(group, 'pred_labels', 'gold_labels')
        functionality_results[name]['precision'] = precision
        functionality_results[name]['recall'] = recall
        functionality_results[name]['f1'] = f1
        functionality_results[name]['accuracy'] = accuracy

    functionality_df = pd.DataFrame.from_dict(functionality_results, orient='index')
    functionality_df = functionality_df.reset_index().rename(columns={"index":"functionality"})

    functionality_df["func_num"] = functionality_df["functionality"].replace(func_num)
    functionality_df["func_num_int"] = functionality_df["functionality"].replace({k: int(v.replace("F", "")) for k, v in func_num.items()})
    functionality_df.to_csv(f"{csv_file}_results.csv")
    print("Results by Functionality:")
    print(functionality_df.sort_values(by="func_num_int", ascending=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CSV file and create result DataFrames.')
    parser.add_argument('--csv-file', type=str, help='Path to the CSV file')

    args = parser.parse_args()

    main(args.csv_file)
