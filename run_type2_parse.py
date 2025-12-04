import os
import re
import json

from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------
# 1. API 설정
# --------------------------------
load_dotenv('.env', override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


# --------------------------------
# 2. PROMPT
# --------------------------------
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


# --------------------------------
# 3. Type2 Parser 실행 함수
# --------------------------------
    
def run_type2_mcq(text):
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


# -------------------------------------
# 4. batch evaluation for type2 parser
# -------------------------------------

def load_text(json_path):
    '''
    input: jsonl 파일 경로 
    output: jsonl의 각 객체의 'text' list
    '''
    ids = []
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            ids.append(item['id'])
            texts.append(item['prompt'])
    return ids, texts

def main(json_path):
    '''
    input: json 파일 경로
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
            parsed_out = run_type2_mcq(text=text)
            
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
    
    return results

# --------------------------------
# 5. save results
# --------------------------------
if __name__ == '__main__':
    json_path="./assets/generation_prompts.json"
    parsed=main(json_path=json_path)
    with open("./results/type2_parsed_results.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
