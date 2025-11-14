import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

celeb_list = ['BrunoMars', 'Dicaprio', 'EunwooCha', 'FanBingbing', 'IshiharaSatomi', 'Jennie', 'JKRowling', 'Obama', 'TaylorSwift', 'TomHolland']

# initialize the heatmap
n_celebs = len(celeb_list)
similarity_matrix = np.zeros((n_celebs, n_celebs))
celeb_to_idx = {celeb: idx for idx, celeb in enumerate(celeb_list)}

# diagonal 
diag_list = []
for celeb in celeb_list:
    try:
        df = pd.read_csv(f'./similarity_results/diag/results_{celeb}.csv')
        avg_sim = df['cosine_similarity'].mean()
        idx = celeb_to_idx[celeb]
        similarity_matrix[idx, idx] = avg_sim
        print(f"✅ {celeb} (diagonal): {avg_sim:.4f}")
        diag_list.append(avg_sim)
    except FileNotFoundError:
        print(f"⚠️ {celeb} 파일을 찾을 수 없습니다.")

# 3. non-diagonal 
non_diag_list = []
for celeb1, celeb2 in combinations(celeb_list, 2):
    try:
        df = pd.read_csv(f'./similarity_results/non_diag/results_{celeb1}_vs_{celeb2}.csv')
        
        avg_sim = df['cosine_similarity'].mean()
        idx1 = celeb_to_idx[celeb1]
        idx2 = celeb_to_idx[celeb2]
       
        similarity_matrix[idx1, idx2] = avg_sim
        similarity_matrix[idx2, idx1] = avg_sim
        
        print(f"✅ {celeb1} vs {celeb2}: {avg_sim:.4f}")
        non_diag_list.append(avg_sim)
    except FileNotFoundError:
        print(f"⚠️ {celeb1} vs {celeb2} 파일을 찾을 수 없습니다.")
        

# draw heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix, 
    annot=True,  # 값 표시
    fmt='.3f',   # 소수점 3자리
    cmap='viridis',  # 색상 (노란색-주황색-빨간색)
    xticklabels=celeb_list,
    yticklabels=celeb_list,
    cbar_kws={'label': 'Cosine Similarity'},
    vmin=0,  # 최소값
    vmax=1,  # 최대값
    square=True,  # 정사각형 셀
    linewidths=0.5,  # 셀 구분선
    linecolor='gray'
)

plt.title('Celebrity Face Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Celebrity', fontsize=12, fontweight='bold')
plt.ylabel('Celebrity', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# save heatmap.png
plt.savefig('./similarity_results/similarity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

diag_avg = np.mean(diag_list)
non_diag_avg = np.mean(non_diag_list)
print(f"⭐️ Average of diagonal values: {diag_avg:.4f}")
print(f"⭐️ Average of non-diagonal values: {non_diag_avg:.4f}")


print("\n" + "="*60)
print("✅ Heatmap 생성 완료!")