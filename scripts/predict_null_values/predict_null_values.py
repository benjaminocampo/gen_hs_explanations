import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI

def extract_info(message):
    if "Label:" in message:
        label_index = message.find("Label:")
        target_index = message.find("Target:")
        explanation_index = message.find("Explanation:")

        label = message[label_index + len("Label:"):target_index].strip()
        target = message[target_index + len("Target:"):explanation_index].strip()
        explanation = message[explanation_index + len("Explanation:"):].strip()

        label = label.split()[0]  # Split and take the first word
        return label, target, explanation
    else:
        is_hateful = "hateful" in message
        against_index = message.find("against")
        because_index = message.find("because")

        if is_hateful:
            target = message[against_index + len("against"):because_index].strip()
            return "hateful", target, message.strip()
        else:
            target = message[message.find("against") + len("against"):because_index].strip()
            explanation = message[because_index + len("because"):].strip()
            return "non-hateful", target, explanation

def build_shot(row):
    shot = (
        f"Message: {row['text']}\n" +
        f"Label: {label_to_str(row['label'])}\n" +
        f"Target: {row['sanitized_target'].lower()}\n" +
        f"Explanation: The message is {label_to_str(row['label'])} because it implies {row['implication']}"
    )
    return shot

def label_to_str(x):
    if x:
        return "hateful"
    else:
        return "non-hateful"

def main():
    parser = argparse.ArgumentParser(description='Evaluate CSV file and create result DataFrames.')
    parser.add_argument('--csv-pred', type=str, help='Path to the CSV file predictions')
    parser.add_argument('--csv-gold', type=str, help='Path to the CSV file gold instances')
    parser.add_argument('--api-key', type=str, help='Model api key')
    args = parser.parse_args()

    openai_client = OpenAI(api_key=args.api_key)

    sbic = pd.read_csv("../../data/sbic_train.csv")
    sbic = sbic[~sbic["implication"].isna()]
    
    csv_pred = pd.read_csv(args.csv_pred)
    csv_gold = pd.read_csv(args.csv_gold)

    # Identify null values from the predictions
    null_indices = csv_pred[
        (csv_pred['pred_label'].isna()) |
        (csv_pred['pred_exp'].isna())
    ].index

    null_indices = [i for i in null_indices if i <= len(csv_gold)]


    # Get missing texts
    texts_to_predict = csv_gold.loc[null_indices, 'text']

    prompt_template = "Given a message: {message}\n1) Label if it is hateful or non-hateful.\n2) Label the target of hate\n3) Generate an explanation of why the sentence is hateful or not.\n\nDemonstration Examples:\n{shots}\nOutput the answer in the following structure.\nLabel:\nTarget:\nExplanation:\n"

    # prompt no shots
    #prompt_template = "Given a message: {message}\n1) Label if it is hateful or non-hateful.\n2) Label the target of hate\n3) Generate an explanation of why the sentence is hateful or not.\nOutput the answer in the following structure.\nLabel:\nTarget:\nExplanation:\n" 
    #engine = "gpt-3.5-turbo"
    engine = "text-davinci-003"
    max_token = 512
    n = 1
    stop = None
    temperature = 0.5
    nof_shots = 5

    inputs = []
    explanations = []
    labels = []
    targets = []
    # Make predictions
    for idx, hateful_message in tqdm(enumerate(texts_to_predict), total=len(texts_to_predict)):

        shots = sbic.sample(n=nof_shots)
        shots = shots.apply(build_shot, axis=1).tolist()
        shots_text = "\n\n".join(shots)

        # Combine the prompt and the hateful message
        input_text = prompt_template.format(message=hateful_message, shots=shots_text)
    
        # Encode the text into tensor of integers using the appropriate tokenizer
        text = openai_client.completions.create(
            model=engine,
            prompt=input_text,
            max_tokens=max_token,
            n=n,
            stop=stop,
            temperature=temperature,
        ).choices[0].text
        
        #text = openai_client.chat.completions.create(
        #    model=engine,
        #    messages=[{"role": "user", "content": input_text}],
        #    max_tokens=max_token,
        #    n=n,
        #    stop=stop,
        #    temperature=temperature,
        #).choices[0].message.content

        label, target, exp = extract_info(text)
        # Generate output from the model
    
        inputs.append(input_text)
        explanations.append(exp)
        labels.append(label)
        targets.append(target)


    csv_pred.loc[null_indices, 'indices'] = null_indices
    csv_pred.loc[null_indices, 'input_text'] = inputs
    csv_pred.loc[null_indices, 'pred_label'] = labels
    csv_pred.loc[null_indices, 'pred_target'] = targets
    csv_pred.loc[null_indices, 'pred_exp'] = explanations

    # Save the DataFrame to a new CSV file
    csv_pred.to_csv(f'inputed_{args.csv_pred}')

if __name__ == "__main__":
    main()
