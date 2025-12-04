import os
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv


# --- API ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


# --- PROMPT ---
SYSTEM_PROMPT = """
You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people are the same person, based on both coarse and fine-grained facial features.
"""

USER_PROMPT = """
[Instruction]
You are given two images: a reference image and a generated image.
Your task is to evaluate whether the two images show the same person based ONLY
on identity-related facial features. Evaluate the Rubrics carefully and follow
the Actions exactly. Do not output anything other than the option number you select.

[Rubrics]
1. Determine whether the two images depict the same person based on:
   • eyes, nose, lips, face shape, skin tone

2. Ignore differences from:
   lighting, color, posture, angle, expression, hairstyle, makeup,
   accessories, image quality.

[Actions]
1. Compare identity-related features.
2. If identity matches → select “Yes”.
3. If any identity feature differs → select one or more “No” options.
4. Return ONLY the numbers (e.g., "1" or "2,4,5").

[Options]
1. Yes, they share the same identity in terms of eyes, nose, lips, skin tone, face shape, and overall facial structure.
2. No, their eyes are quite different.
3. No, their noses are quite different.
4. No, their lips are quite different.
5. No, their face shapes are quite different.
6. No, their skin tones are quite different.
7. No, the overall facial structure and spatial arrangement of their features are quite different.
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
    

# find ref-gen match
VALID_EXT = [".jpg", ".jpeg", ".png"]
def find_matching_gen(gen_folder, idx):
    for ext in VALID_EXT:
        candidate = os.path.join(gen_folder, idx + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# --- Type1_MCQ ---
def run_type1_mcq(ref_img_path, gen_img_path):
    print("run_type1_mcq called:", ref_img_path, gen_img_path)

    ref_b64 = encode_image(ref_img_path)
    gen_b64 = encode_image(gen_img_path)

    ref_mime = get_mime(ref_img_path)
    gen_mime = get_mime(gen_img_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{ref_mime};base64,{ref_b64}"
                        }
                    },
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



# --- Run Type1_MCQ evaluation ---
def main(ref_folder_path, gen_folder_path, output_path):
    ref_folder = ref_folder_path
    gen_folder = gen_folder_path

    results = {}

    for fname in sorted(os.listdir(ref_folder)):
        ext = os.path.splitext(fname)[1].lower()

        if ext not in VALID_EXT:
            continue

        idx = os.path.splitext(fname)[0]   # "001"
        ref_path = os.path.join(ref_folder, fname)

        gen_path = find_matching_gen(gen_folder, idx)

        if gen_path is None:
            print(f"[{idx}] No matching generated file (jpg/jpeg/png) → skip")
            continue

        print(f"[{idx}] Evaluating...")

        try:
            mcq_out = run_type1_mcq(ref_path, gen_path)

            print(f"[{idx}] → {mcq_out}")

            results[idx] = mcq_out

        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
            results[idx] = "ERROR"

    # save results
    output_list = []

    for idx, mcq_out in results.items():
        output_list.append({
            "id": idx.zfill(3), 
            "result": mcq_out
        })

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved → {output_path}")


if __name__ == "__main__":
    ref_folder_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/reference"
    gen_folder_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/generation/generated_gpt5.1"
    output_path = "type1_results.json"
    main(ref_folder_path, gen_folder_path, output_path)
