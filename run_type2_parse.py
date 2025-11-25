import os
from openai import OpenAI
import json
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
    print("run_type2_mcq called:", text)
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

def load_text(jsonl_path):
    with open(jsonl_path, 'r') as f:
        texts = []
        for line in f:
            data = json.loads(line)
            texts.append(data['text_prompt'])
    return texts

def main(jsonl_path):
    texts = load_text(jsonl_path=jsonl_path)
    results = {}
    
    for i, text in enumerate(texts):
        if text is None:
            print(f"{i}-th text is missing.")
            continue
        
        print(f"{i}-th text is being parsed...")
        
        try:
            parsed_out = run_type2_mcq(text=text)
            print(f"{i}-th text parsed_out: {parsed_out}")
            results[i] = parsed_out
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            results[i] = {"error": str(e), "text": text} 
    
    return results

# --------------------------------
# 5. save results
# --------------------------------
if __name__ == '__main__':
    jsonl_path="./test_prompts.jsonl"
    parsed=main(jsonl_path=jsonl_path)
    with open("./type2_parsed_results.json", "w") as f:
        json.dump(parsed, f, indent=2)

