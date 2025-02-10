import os
import openai
import json
import time

openai.api_type = "azure"
openai.api_base = "your api base here"
openai.api_version = "your version"
openai.api_key = "your api key here"

system_message = {"role": "system", "content": "You are a medical AI trained to identify and classify tokens into three categories: Clinical Impacts, Social Impacts, and Outside ('O'). The data you are working with has been collected from 14 forums on Reddit (subreddits) that focused on prescription and illicit opioids, and medications for opioid use disorder. This dataset represents a social media context, coming from individuals who may use prescription and illicit opioids and stimulants. In this dataset, high-frequency clinical impacts include 'withdrawal', 'rehab', 'addicted', 'detox', 'overdosed', and 'rehabs'. High-frequency social impacts include 'lost', 'homeless', 'charged', 'streets', 'jail', and 'disorderly'. Your task is to extract and classify the clinical and social impacts from this dataset, considering your knowledge of the lifestyle of this population and the potential clinical and social impacts they might experience. 'Clinical Impacts' refer to tokens describing the effects, consequences, or impacts of substance use on individual health or well-being, as defined in UMLS. 'Social Impacts' describe the societal, interpersonal, or community-level effects, also based on UMLS definitions. Any token not falling into these categories should be labeled as 'O'. Your task is to predict and return the label for each provided token, ensuring the number of output labels matches the number of input tokens exactly. The output format should be tokens with their labels: ['I-O', 'was-O', 'a-O', 'codeine-Clinical Impacts', 'addict-Clinical Impacts', '.-O']. Please strictly follow the output format to output the results. Please be sure to output each token and its corresponding label, and link them with '-'. Please use the tokens I provided, do not use other tokenization mechanism. Possible analysis of prediction errors: If a sentence describes the background information of an event, facility, or project, then even if it mentions keywords related to social impact like 'at jail', it still cannot be determined as describing a patient being in jail. It is essential to clearly determine whether the sentence is describing the patient's condition. Second, if the sentence is about the usage, operation, or introduction of a drug or medicine, it does not belong to the patient's clinical impacts, even if it mentions some symptoms. Pay attention to whether the sentence contains words like 'if' that indicate conditions."}


# Helper function to process the best match data and interleave 'u' and 'a'
def process_best_matches(matches):
    interleaved_messages = []
    for i, match in enumerate(matches):
        tokens = match["Tokens"]
        labels = match["Labels"]
        
        # Ensure that 'tokens' and 'labels' are strings with correct formatting
        assistant_message_u_content = f"{tokens}"  # Directly use tokens, no additional formatting
        assistant_message_u = {"role": "user", "content": assistant_message_u_content}

        assistant_message_a_content = f"{labels}"  # Directly use labels, no additional formatting
        assistant_message_a = {"role": "assistant", "content": assistant_message_a_content}

        # Append 'u' followed by its corresponding 'a'
        interleaved_messages.append(assistant_message_u)
        interleaved_messages.append(assistant_message_a)

    return interleaved_messages

file_path = '/Users/yaoge/Downloads/impacts_tokens_with_updated_labels_top_10_dpr.txt'

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
        
        # Find the original sentence
        original_sentence = None
        best_matches = []
        for line in lines:
            if line.startswith("Original Sentence:"):
                original_sentence = safe_extract(line)  # No need to split; keep the raw format
            elif line.startswith("Best Match:"):
                tokens_line = lines[lines.index(line) + 1]  # Line after 'Best Match' contains tokens
                labels_line = lines[lines.index(line) + 2]  # Line after tokens contains labels
                
                # Process tokens and labels
                tokens = safe_extract(tokens_line)
                labels = safe_extract(labels_line)
                
                best_matches.append({"Tokens": tokens, "Labels": labels})
        
        if original_sentence and best_matches:
            # User message with the original sentence
            user_message = {"role": "user", "content": original_sentence}
            
            # Generate assistant messages (u and a interleaved)
            interleaved_messages = process_best_matches(best_matches)
            
            # Construct messages for the current iteration (system message followed by user, u-a pairs)
            messages = [system_message] + interleaved_messages + [user_message]

            # Ensure JSON format is valid (double quotes for keys)
            #print(json.dumps(messages, indent=4))

            #print(messages)
            #break

            # Make the API call
            completion = openai.ChatCompletion.create(
            engine="app1008",
            messages=messages,
            temperature=0.2,
            max_tokens=4000,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
            
            # Process the response
            with open('/Users/yaoge/Downloads/test_preds_impacts_gpt4_RAG_5shot_DPR.txt','a') as wf:
                print(user_message)
                wf.write(completion['choices'][0]['message']['content'])
            #print(completion['choices'][0]['message']['content'])

            # Add a delay if needed to avoid rate limits
            time.sleep(60)
