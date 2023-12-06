import argparse
import pandas as pd
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score

def calculate_metrics(data, functionality=None):
    metrics = [("bertscore","bertscore"), ("bleurt", "bleurt"), ("bleu","bleu"), ("bleu", "bleu_IvsP")]
    params = [{"model_type": "distilbert-base-uncased"}, {}, {"max_order": 1, "smooth": True}, {"max_order": 1, "smooth": True}]
    val_to_output = ["f1", "scores", "bleu", "bleu"]
    pred_ref = [("pred_exp", "gold_exp"), ("pred_exp", "gold_exp"), ("pred_exp", "gold_exp"), ("pred_exp", "text")]
    results = {}
    for (metric_to_load, metric_name), ps, out, (pred_col, ref_col) in zip(metrics, params, val_to_output, pred_ref):
        metric = evaluate.load(metric_to_load)
        if functionality:
            grouped_data = data.groupby(functionality)
            scores_by_group = {}
            for name, group in grouped_data:
                scores = metric.compute(predictions=group.reset_index()[pred_col], references=group.reset_index()[ref_col], **ps)
                scores_by_group[name] = np.mean(scores[out])
            results[metric_name] = scores_by_group
        else:
            scores = metric.compute(predictions=data[pred_col], references=data[ref_col], **ps)
            results[metric_name] = np.mean(scores[out])

    return results

def main(csv_pred, csv_gold, prefix_out_filename):

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
    data_gold = pd.read_csv(csv_gold)

    data_pred = pd.read_csv(csv_pred)

    data = pd.concat([data_gold, data_pred], axis=1)

    data = data[~data["functionality"].isna()]
    
    data = data[~data["functionality"].apply(lambda t: t.startswith("spell_"))]

    assert len(data) == 2968, "Wrong total number of instances in HC. Some instance is being filtered."

    data["func_num"] = data["functionality"].replace(func_num)

    # We are removing null pred and exp values

    data = data[~data["pred_label"].isna()]
    data = data[~data["pred_exp"].isna()]

    print("Null values that we are not considering: ", 2968 - len(data))

    data["pred_label"] = data["pred_label"].apply(lambda t: t.lower())
    data["pred_label"] = data["pred_label"].replace({"hateful": 1, "non-hateful": 0})
    data["gold_label"] = data["gold_label"].replace({"hateful": 1, "non-hateful": 0})

    import pdb; pdb.set_trace()
    # DataFrame for overall results
    overall_results = {}

    # Evaluate generations overall and store results
    overall_results = calculate_metrics(data)

    # Classification metrics overall and store results
    accuracy = accuracy_score(data['pred_label'], data['gold_label'])
    overall_results['accuracy'] = accuracy

    overall_df = pd.DataFrame([overall_results])
    print("Overall Results:")
    print(overall_df)
    overall_df.to_csv(f"{prefix_out_filename}_overall_results.csv", index=False)

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
        accuracy = accuracy_score(group['pred_label'], group['gold_label'])
        functionality_results[name]['accuracy'] = accuracy

    functionality_df = pd.DataFrame.from_dict(functionality_results, orient='index')
    functionality_df = functionality_df.reset_index().rename(columns={"index":"functionality"})

    functionality_df["func_num"] = functionality_df["functionality"].replace(func_num)
    functionality_df["func_num_int"] = functionality_df["functionality"].replace({k: int(v.replace("F", "")) for k, v in func_num.items()})
    print("Results by Functionality:")
    print(functionality_df.sort_values(by="func_num_int"))
    functionality_df.to_csv(f"{prefix_out_filename}_func_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CSV file and create result DataFrames.')
    parser.add_argument('--csv-pred', type=str, help='Path to the CSV file predictions')
    parser.add_argument('--csv-gold', type=str, help='Path to the CSV file gold instances')
    parser.add_argument('--prefix-out-filename', type=str, help='Prefix of output files')

    args = parser.parse_args()

    main(args.csv_pred, args.csv_gold, args.prefix_out_filename)
