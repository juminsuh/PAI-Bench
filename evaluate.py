from metric import *

def main():

    path = "./preliminary/mcq/neg.json"
    # path = "./preliminary/neg/fgis_similarity.csv"

    filename = os.path.basename(path)         
    ext = os.path.splitext(filename)[1]
    type = os.path.splitext(filename)[0]
    
    if ext == ".json":
        choices, num_factors = load_json(json_path=path)
        if num_factors == []: # type1
            score = evaluate_type1(choices=choices)
            print("type1")
        else: # type2
            score = evaluate_type2(choices=choices, num_factors=num_factors)
            print("type2")
        print(f"✅ Score for {type}: {score:.6f}")
    elif ext == ".csv":
        score = compute_avg(path)
        print(f"✅ Score for {type}: {score:.6f}")
    
if __name__ == '__main__':
    main()