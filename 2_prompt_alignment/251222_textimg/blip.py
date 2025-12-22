"""
Compute BLIPScore using BLIP Image-Text Matching (ITM) head
- Images are stored as {id}.jpg
- Prompts are loaded from generation_prompts.json
"""

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

from lavis.models import load_model_and_preprocess


def load_blip_itm(device="cuda"):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_itm",
        model_type="base",
        is_eval=True,
        device=device
    )
    return model, vis_processors, txt_processors

# load prompts from json
def load_prompts(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    prompt_dict = {}
    for item in data:
        prompt_dict[item["id"]] = item["prompt"]

    return prompt_dict


def load_image(image_path, vis_processor):
    image = Image.open(image_path).convert("RGB")
    return vis_processor["eval"](image).unsqueeze(0)


@torch.no_grad()
def compute_blip_itm_score(model, image, text):
    """
    Returns:
      - match logit
      - match probability
    """
    output = model(
        {
            "image": image,
            "text_input": text
        },
        match_head="itm"
    )

    # output shape: [1, 2] → [non-match, match]
    logit = output[:, 1]
    prob = torch.softmax(output, dim=-1)[:, 1]

    return {
        "logit": logit.item(),
        "probability": prob.item()
    }


def calculate_blip_scores(
    image_dir: str,
    prompt_json: str,
    output_path: str,
    device: str = "cuda"
):
    image_dir = Path(image_dir)

    # Load model
    model, vis_processors, txt_processors = load_blip_itm(device)

    # Load prompts
    prompt_dict = load_prompts(prompt_json)

    image_files = sorted(image_dir.glob("*.jpg"))

    results = {}
    failed = []

    for img_path in tqdm(image_files, desc="Computing BLIPScore (ITM)"):
        img_id = img_path.stem  # "001", "002", ...

        if img_id not in prompt_dict:
            failed.append(img_id)
            continue

        try:
            image = load_image(img_path, vis_processors).to(device)
            text = txt_processors["eval"](prompt_dict[img_id])

            score = compute_blip_itm_score(model, image, text)
            results[img_id] = {
                "prompt": prompt_dict[img_id],
                **score
            }

        except Exception as e:
            print(f"[ERROR] {img_id}: {e}")
            failed.append(img_id)

   

    # stats
    probs = [v["probability"] for v in results.values()]

    stats = {
        "num_samples": len(results),
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "failed": len(failed)
    }

    output = {
        "metric": "BLIPScore_ITM",
        "scores": results,
        "statistics": stats,
        "failed_ids": failed
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nBLIPScore (ITM) finished")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    calculate_blip_scores(
        image_dir="/data2/jiyoon/PAI-Bench/data/251222/cropped/generated_gemini2.5Flash",
        prompt_json="/data2/jiyoon/PAI-Bench/data/datasets_final/generation_prompts.json",
        output_path="/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_textimg/results/blip_itm.json",
        device="cuda"
    )
