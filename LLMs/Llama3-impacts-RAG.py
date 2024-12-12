import requests
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama3 API and model details
url = 'http://llm1.priv.bmi.emory.edu:8000/llama3_70B/v1/completions'
model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# System message remains the same
system_message = {
    "role": "system",
    "content": "You are a medical AI trained to identify and classify tokens into three categories: Clinical Impacts, Social Impacts, and Outside ('O'). The data you are working with has been collected from 14 forums on Reddit (subreddits) that focused on prescription and illicit opioids, and medications for opioid use disorder. This dataset represents a social media context, coming from individuals who may use prescription and illicit opioids and stimulants. Your task is to extract and classify the clinical and social impacts from this dataset. The output format should be tokens with their labels: ['I-O', 'was-O', 'a-O', 'codeine-Clinical Impacts', 'addict-Clinical Impacts', '.-O']."
}

# Helper function to process the best match data and interleave 'u' and 'a'
def process_best_matches(matches):
    interleaved_messages = []
    for i, match in enumerate(matches):
        tokens = match["Tokens"]
        labels = match["Labels"]
        
        assistant_message_u_content = f"{tokens}"  # Directly use tokens
        assistant_message_u = {"role": "user", "content": assistant_message_u_content}

        assistant_message_a_content = f"{labels}"  # Directly use labels
        assistant_message_a = {"role": "assistant", "content": assistant_message_a_content}

        # Append 'u' followed by its corresponding 'a'
        interleaved_messages.append(assistant_message_u)
        interleaved_messages.append(assistant_message_a)

    return interleaved_messages

file_path = 'impacts_tokens_with_updated_labels_top_10_dpr.txt'

# Function to safely extract the content after ':', avoiding eval
def safe_extract(line):
    try:
        # Strip leading/trailing whitespace without modifying the content format
        content = line.split(":", 1)[1].strip()
        return content
    except IndexError:
        return ""

# Read the file and process the data
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    data = content.split("==================================================")  # Separate by original sentence blocks
    
    for block in data:
        lines = block.strip().splitlines()
        if not lines:
            continue
        
        original_sentence = None
        best_matches = []
        for line in lines:
            if line.startswith("Original Sentence:"):
                original_sentence = safe_extract(line)  # Keep the raw format
            elif line.startswith("Best Match:"):
                tokens_line = lines[lines.index(line) + 1]  # Line after 'Best Match' contains tokens
                labels_line = lines[lines.index(line) + 2]  # Line after tokens contains labels
                
                tokens = safe_extract(tokens_line)
                labels = safe_extract(labels_line)
                
                best_matches.append({"Tokens": tokens, "Labels": labels})
        
        if original_sentence and best_matches:
            user_message = {"role": "user", "content": original_sentence}
            
            interleaved_messages = process_best_matches(best_matches)
            
            # Construct messages (system message followed by user, interleaved u-a pairs)
            messages = [system_message] + interleaved_messages + [user_message]
            
            # Tokenize and format the messages for Llama3
            messages_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # API call data
            data = {
                "model": model_name,
                "prompt": messages_text,
                "max_tokens": 800,
                "temperature": 0.5,
                "top_p": 0.95
            }

            # Make the API call to Llama3
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=data)

            if response.status_code == 200:
                completion = response.json()['choices'][0]['text']

                with open('test_preds_llama3_5shot_DPR.txt', 'a') as wf:
                    print(user_message)
                    #print('answer=====', completion)
                    wf.write(completion[46:])
                
            else:
                print(f"Error: {response.status_code}, {response.text}")
            
