from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import torch
import pandas as pd

ihc_is_train = pd.read_csv("../../data/ihc_is_train.csv")
ihc_is_dev = pd.read_csv("../../data/ihc_is_dev.csv")

# GPU out of memory, we are trying with samples
ihc_is_train = ihc_is_train.sample(50, random_state=0)
ihc_is_dev = ihc_is_dev.sample(10, random_state=0)

exp_prompt = "Given a message: {message}\n1) Label if it is hateful or non-hateful.\n2) Label the target of hate\n3) Generate an explanation of why the sentence is hateful or not.\nOutput the answer in the following structure.\nLabel:\nTarget:\nExplanation:\n"
out_exp = "Label: {label}\nTarget: {target}\nExplanation: {gold_exp}"

ihc_is_train["input_prompt"] = ihc_is_train["text"].apply(lambda t: exp_prompt.format(message=t))
ihc_is_dev["input_prompt"] = ihc_is_dev["text"].apply(lambda t: exp_prompt.format(message=t))
ihc_is_train["output_prompt"] = ihc_is_train["gold_exp"].apply(lambda t: exp_prompt.format(message=t))
ihc_is_dev["output_prompt"] = ihc_is_dev["gold_exp"].apply(lambda t: exp_prompt.format(message=t))


ihc_is_train_dataset = Dataset.from_pandas(ihc_is_train)
ihc_is_dev_dataset = Dataset.from_pandas(ihc_is_dev)


tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    # Assuming examples is a dict with 'messages' and 'explanations' as keys
    model_inputs = tokenizer(examples['input_prompt'], padding="max_length", truncation=True, max_length=256)
    # Tokenize the explanations and add them as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['output_prompt'], padding="max_length", truncation=True, max_length=256)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the tokenization to your dataset here
# This is an example; adjust it to how your dataset is loaded (e.g., using datasets library)
tok_train = ihc_is_train_dataset.map(tokenize_function, batched=True, remove_columns=ihc_is_train_dataset.column_names)
tok_dev = ihc_is_dev_dataset.map(tokenize_function, batched=True, remove_columns=ihc_is_train_dataset.column_names)


model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Assuming your model and tokenizer are already instantiated
tokenizer.pad_token = tokenizer.eos_token  # Make sure the pad token is set

# Instantiate the custom data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Continue setting up your TrainingArguments as before
training_args = TrainingArguments(
    output_dir='./mistral-7B_ft_with_exp/results',
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    per_device_eval_batch_size=4,
    logging_dir='./mistral-7B_ft_with_exp/logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    eval_accumulation_steps=16,
    no_cuda=False,
)

# Update the Trainer to use the custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_dev,
    data_collator=data_collator,
)

trainer.train()

trainer.push_to_hub(repo_id="BenjaminOcampo/mistral-7B_ft_with_exp",
                    token="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",
                    private=True)
trainer.save_model("./mistral-7B_ft_with_exp")
