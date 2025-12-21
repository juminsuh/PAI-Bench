import os
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv


# --- API ---
load_dotenv('.env', override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


# --- PROMPT ---
SYSTEM_PROMPT = """
You are a precise vision–language evaluator who inspects whether the text description matches the content in the image based on the specified factors. 
"""

USER_PROMPT = """
[Instruction]
You are given:
	• one image
	• a set of style/action factors: {factors}
	• and a text description: {description}
Your task is to evaluate whether the description correctly matches the visual content of the image specifically in terms of the provided factors. Follow the Rubrics and Actions exactly as written.


[Rubrics]
1.  Factors may include the following elements:
	• action
	• hairstyle 
	• emotion (e.g., smiling, not smiling, frowning)
	• makeup
	• clothes
	• accessories (e.g., hat, earrings)
2.  Evaluate the image based on the factors included in {factors}.  Only the listed factors should influence your decision.
3.  Ignore any attributes not included in the factor list. If a factor is not included in {factors}, you must not judge it.

 
[Actions]
1. Compare the description **with the image**, evaluating only the factors listed in {factors}. Do not make judgments on any attributes outside these factors.
2. Choose the correct answer from the Options section:
	• If all provided factors match the image → select the option that starts with "Yes".
	• If any provided factor does not match the image → select one or more options that start with "No" corresponding to the mismatched factor(s).
    • You may select multiple options, but **you must not choose an option starting with "Yes" together with any option starting with "No."**
3. Return **only the option number(s)**:
	• If only one option is selected → return a single number (e.g., 1).
	• If more than one options are selected  → return all option numbers separated by commas (e.g., 2, 4, 5)
4. Do not output any explanations, text, or the option sentences. Return only the number(s).


[Options]
{options}
"""


# --- Utils ---
# encode img
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    else:
        return "image/jpeg"
    
VALID_EXT = [".jpg", ".jpeg", ".png"]


# --- Type2_MCQ ---
# generate MCQ options 
def generate_options(factors):
    options = """1. Yes, the description accurately describes the image content in terms of all provided factors.\n"""
    for idx, factor in enumerate(factors):
        options += f"{idx + 2}. No, the {factor} of the image does not match the description.\n"
    return options


def run_type2_mcq(gen_img_path, factors, description, options):
    print("run_type2_mcq called:", gen_img_path)

    gen_b64 = encode_image(gen_img_path)
    gen_mime = get_mime(gen_img_path)

    formatted_prompt = USER_PROMPT.format(factors=factors, description=description, options=options)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{gen_mime};base64,{gen_b64}"
                        }
                    },
                ]
            },
        ],
    )

    return response.choices[0].message.content.strip()



# --- Run Type2 MCQ evaluation ---
def main(gen_folder_path, parsed_data_path, output_path):
    gen_folder = gen_folder_path
    
    # load parsed data 
    with open(parsed_data_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    # create lookup dict for quick access
    parsed_lookup = {item['id']: item for item in parsed_data}


    results = {}

    for fname in sorted(os.listdir(gen_folder)):
        ext = os.path.splitext(fname)[1].lower()

        if ext not in VALID_EXT:
            continue

        idx = os.path.splitext(fname)[0]   # "001"
        gen_path = os.path.join(gen_folder, fname)

        # check if parsed data exists for this img
        if idx not in parsed_lookup:
            print(f"[{idx}] No parsed data found → skip")
            continue

        parsed_item = parsed_lookup[idx]
        factors = parsed_item['factor']
        description = parsed_item['description']
        
        # generate options based on factors
        options = generate_options(factors)
        print(options)

        print(f"➡️ [{idx}] Evaluating...")

        try:
            mcq_out = run_type2_mcq(gen_path, factors, description, options)
            results[idx] = {
                "result": mcq_out,
                "num_factors": len(factors)
            }
            print("✅", results[idx])
            print()

        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
            results[idx] = {
                "result": "ERROR",
                "num_factors": len(factors) if 'factors' in locals() else 0
            }

    # save results
    output_list = []

    for idx, result_data in results.items():
        if isinstance(result_data, dict):
            output_list.append({
                "id": idx.zfill(3), 
                "result": result_data["result"],
                "num_factors": result_data["num_factors"]
            })
        else:
            # Handle legacy format if any
            output_list.append({
                "id": idx.zfill(3), 
                "result": result_data,
                "num_factors": 0
            })


    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved → {output_path}")


if __name__ == "__main__":
    gen_folder_path = "./generation/generated_gpt5.1"
    parsed_data_path = "./results/parsed_results.json"
    output_path = "./results/type2/type2_mcq_gpt5.1.json"
    main(gen_folder_path, parsed_data_path, output_path)
