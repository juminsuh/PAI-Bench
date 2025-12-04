import re
import os
import json
import numpy as np
import pandas as pd

def load_json(json_path):
    '''
    input: jsonl 파일 경로 
    output: jsonl의 각 객체의 'result' list
    '''
    choices = []
    num_factors = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            choices.append(item['result'])
            if "type2" in json_path:
                num_factors.append(item['num_factors'])
                
    return choices, num_factors

def compute_avg(csv_path):
    df = pd.read_csv(csv_path)
    avg = df['similarity'].mean()
    return avg

def evaluate_type1(choices):
    scores = []
    for choice in choices:
        score = 1
        if choice == "1":
            scores.append(score)
        else:
            # parse choice
            numbers = re.findall(r"\d+", choice)
            score -= len(numbers)*(1/6)
            scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score

def evaluate_type2(choices, num_factors):
    scores = []
    for choice, num_factor in zip(choices, num_factors):
        score = 1
        if choice == "1":
            scores.append(score)
        else:
            # parse choice
            numbers = re.findall(r"\d+", choice)
            score -= len(numbers)*(1/(num_factor+1))
            scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score