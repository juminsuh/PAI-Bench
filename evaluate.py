from utils import *

def main():

    path = "/home/jiyoon/PAI-Bench/experiment_1/mcq_type1/results/neg.json"
    out_path = "/home/jiyoon/PAI-Bench/experiment_1/mcq_type1/scores/neg.json"

    filename = os.path.basename(path)         
    ext = os.path.splitext(filename)[1]
    type = os.path.splitext(filename)[0]


    # --- MCQ scoring ---
    if ext == ".json":
        items, num_factors = load_json(json_path=path)

        # type1
        if num_factors == []: 
            avg_score, results = evaluate_type1(items=items)
            print("type1")

        # type2
        else: 
            avg_score, results = evaluate_type2(items=items, num_factors=num_factors)
            print("type2")

        # print results
        print(f"Avg score for {type}: {avg_score:.6f}")
        
        # save results
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed scores saved to: {out_path}")
    

    # --- Emb based metric ---
    elif ext == ".csv":
        score = compute_avg(path)
        print(f"Score for {type}: {score:.6f}")
    

if __name__ == '__main__':
    main()