import os
import re
import json

from openai import OpenAI
from dotenv import load_dotenv


# --- API ---
load_dotenv('.env', override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


# --- PROMPT ---
SYSTEM_PROMPT = f"""
You are a powerful language parser capable of accurately analyzing prompts, identifying which factors they describe, and rewriting them into a clean, concise visual description.
"""

USER_PROMPT = """
[Instruction]
You are given a raw text prompt. Your task is to:
1. Identify which style/action factors the prompt describes, and
2. Rewrite the prompt into a clean, concise visual description that includes only the content relevant to those extracted factors.
When performing this task, carefully follow the Rubrics below and apply the Actions exactly as written.

[Rubrics]
1. You must determine which of the following style/action factors are explicitly expressed in the prompt:
	• action
	• hairstyle 
	• emotion
	• makeup
	• clothes
	• accessories
    You may extract one or more factors (between 1 and 6), as long as they appear explicitly in the prompt. Do not add factors outside this list.
2. Do not include identity-related factors such as eyes, nose, lips, face shape, and skin tone. These must not be extracted as style/action factors and must not appear in the rewritten description.
3. The rewritten description should:
	• remove meta-phrases such as “generate an image of”, “a photo of”, etc.
	• keep only the elements directly related to the extracted factors
    • avoid adding any new details not present in the original prompt
	• be short, clean, and caption-like

[Actions]
1. Extract the style/action factors present in the prompt. Multiple factors may be extracted if the prompt contains more than one. Only select from the six allowed categories.
2. Rewrite the prompt into a concise description that uses only the extracted factors and contains no additional invented details.
3. Return the final answer in the following format:
	Factors: [factor_1, factor_2, …]
	Description: rewritten_visual_description

[Examples]
These are some examples which complete given task.
- Example 1)
   • text prompt: "Generate an image of this man wearing a yellow shirt."
   • Factors: [clothes]
   • Description: "man wearing a yellow shirt"
- Example 2)
   • text prompt: "Generate an image of this girl smiling in front of the christmas tree."
   • Factors: [emotion, action]
   • Description: "girl smiling in front of the christmas tree"

Now analyze the following prompt: {description}
"""

# --- Utils ---
def load_text(json_path):
    '''
    input: jsonl file path
    output: 'text' list of each object in jsonl file
    '''
    ids = []
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            ids.append(item['id'])
            texts.append(item['prompt'])
    return ids, texts


# --- Type2_Parse ---
def run_type2_parse(text):
    formatted_prompt = USER_PROMPT.format(description=text)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=50, # increase max_tokens
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt}
                ]
            },
        ],
    )

    return response.choices[0].message.content.strip()



# --- Run Type2 Parser ---
def main(json_path, output_path):
    '''
    input: json file path
    output: json file 
    '''
    ids, texts = load_text(json_path=json_path)
    results = []
    
    factors_pattern = r"Factors:\s*\[([^\]]+)\]"
    desc_pattern = r'Description:\s*"?([^"\n]+)"?'
    
    for id, text in zip(ids, texts):
        if text is None:
            print(f"{id} text is missing.")
            continue
        
        print(f"📌 Parsing {id}...")
        
        try:
            parsed_out = run_type2_parse(text=text)
            
            factors = re.search(factors_pattern, parsed_out).group(1)
            factor_list = [x.strip() for x in factors.split(",")]
            description = re.search(desc_pattern, parsed_out).group(1)
            
            print(f"\n✅ {id} text have {factor_list}")
            print(f"✅ {id} text parsed_out: {text} ➡️ {description}")
            
            results.append({
                'id': id,
                'factor': factor_list,
                'description': description
            })
        except Exception as e:
            print(f"[{id}] ERROR: {e}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Saved → {output_path}")


if __name__ == '__main__':
    json_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/generation_prompts.json"
    output_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/type2_parsed_results.json"

    main(json_path, output_path)
