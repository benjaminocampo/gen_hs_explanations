from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType
from peft import get_peft_model
import torch
import pandas as pd

ihc_is_train = pd.read_csv("../data/ihc_is_train.csv")
ihc_is_dev = pd.read_csv("../data/ihc_is_dev.csv")

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

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Assuming your model and tokenizer are already instantiated
tokenizer.pad_token = tokenizer.eos_token  # Make sure the pad token is set

# Instantiate the custom data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Continue setting up your TrainingArguments as before
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=32,  # Adjust based on your GPU memory
    per_device_eval_batch_size=32,
    logging_steps=50,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #optim="adafactor",
    #gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    #gradient_checkpointing=True,
    #no_cuda=False,
    output_dir='./mistral-7B_ft_with_exp/results',
    logging_dir='./mistral-7B_ft_with_exp/logs',
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

trainer.push_to_hub("BenjaminOcampo/mistral-7B_ft_with_exp")
trainer.save_model("./mistral-7B_ft_with_exp")
