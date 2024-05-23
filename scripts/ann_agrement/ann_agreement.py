from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
import krippendorff

ann1 = pd.read_csv("./implicit_annotation_ann1_hatecheck.csv")
ann2 = pd.read_csv("./implicit_annotation_ann2_hatecheck.csv")

ann2 = ann2[ann1["keep_hate"] == 1]
ann1 = ann1[ann1["keep_hate"] == 1]

ann1 = ann1.rename(columns={"gold_exp_similarity": "similarity", "input_dissimilarity": "originality"})
ann2 = ann2.rename(columns={"gold_exp_similarity": "similarity", "input_dissimilarity": "originality"})

reliability_data_sim = np.vstack([ann1["similarity"], ann2["similarity"]])
reliability_data_org = np.vstack([ann1["originality"], ann2["originality"]])
reliability_data_cont = np.vstack([ann1["context"], ann2["context"]])

k_alpha_sim = krippendorff.alpha(reliability_data=reliability_data_sim, level_of_measurement="ordinal")
k_alpha_org = krippendorff.alpha(reliability_data=reliability_data_org, level_of_measurement="ordinal")
k_alpha_cont = krippendorff.alpha(reliability_data=reliability_data_cont, level_of_measurement="ordinal")
percent_sim = (ann2["similarity"] == ann1["similarity"]).sum() / len(ann2)
percent_org = (ann2["originality"] == ann1["originality"]).sum() / len(ann2)
percent_cont = (ann2["context"] == ann1["context"]).sum() / len(ann2)

print("k_alpha_sim:", k_alpha_sim)
print("k_alpha_org:", k_alpha_org)
print("k_alpha_cont:", k_alpha_cont)

print("percent_sim:", percent_sim)
print("percent_org:", percent_org)
print("percent_cont:", percent_cont)

print("ann1_mean_sim:", ann1["similarity"].mean())
print("ann1_mean_org:", ann1["originality"].mean())
print("ann1_mean_cont:", ann1["context"].mean())

print("ann1_std_sim:", ann1["similarity"].std())
print("ann1_std_org:", ann1["originality"].std())
print("ann1_std_cont:", ann1["context"].std())

print("ann2_mean_sim:", ann2["similarity"].mean())
print("ann2_mean_org:", ann2["originality"].mean())
print("ann2_mean_cont:", ann2["context"].mean())

print("ann2_std_sim:", ann2["similarity"].std())
print("ann2_std_org:", ann2["originality"].std())
print("ann2_std_cont:", ann2["context"].std())
