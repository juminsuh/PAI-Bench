import re
import os
import json
import numpy as np
import pandas as pd

def load_json(json_path):
    num_factors = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if "type2" in json_path:
                num_factors.append(item['num_factors'])
                
    return data, num_factors


def normalize(emotion, metric):
    
    df = pd.read_csv(f'./preliminary/{emotion}/{metric}_similarity.csv')

    # min-max scaling -> fit range [0, 1]
    min_val = df['similarity'].min()
    max_val = df['similarity'].max()

    df['similarity_normalized'] = (df['similarity'] - min_val) / (max_val - min_val)
    df = df.drop('similarity', axis=1)

    df.to_csv(f'./preliminary/normalized/{emotion}/{metric}_similarity.csv', index=False)


def compute_avg(csv_path):
    df = pd.read_csv(csv_path)
    avg = df['similarity'].mean()
    return avg


def evaluate_type1(items):
    scores = []
    results = []
    for item in items:
        choice = item['result']
        score = 1
        if choice == "1":
            scores.append(score)
        else:
            numbers = re.findall(r"\d+", choice)
            score -= len(numbers)*(1/6)
            scores.append(score)
        
        # save results for neg pairs
        if 'ref_id' in item and 'gen_id' in item: 
            result_entry = {
                'ref_id': item['ref_id'],
                'gen_id': item['gen_id'],
                'result': choice,
                'score': score
            }

        # save results for pos pairs
        elif 'id' in item:
            result_entry = {
                'id': item['id'],
                'result': choice,
                'score': score
            }
        
        results.append(result_entry)
    
    avg_score = np.mean(scores)
    return avg_score, results

def evaluate_type2(items, num_factors):
    scores = []
    results = []
    for item, num_factor in zip(items, num_factors):
        choice = item['result']
        score = 1
        if choice == "1":
            scores.append(score)
        else:
            numbers = re.findall(r"\d+", choice)
            score -= len(numbers)*(1/(num_factor+1))
            scores.append(score)
    
        # save results for neg pairs
        if 'ref_id' in item and 'gen_id' in item:
            result_entry = {
                'ref_id': item['ref_id'],
                'gen_id': item['gen_id'],
                'result': choice,
                'score': score
            }

        # save results for pos pairs
        elif 'id' in item:
            result_entry = {
                'id': item['id'],
                'result': choice,
                'score': score
            }
        
        results.append(result_entry)
    
    avg_score = np.mean(scores)
    return avg_score, results