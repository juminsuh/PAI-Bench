import os
import random
from pathlib import Path

image_folder = "./assets/images/TomHolland"  # 폴더 경로 지정

# 이미지 파일 확장자
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

# 폴더 내 모든 이미지 파일 가져오기
image_files = [
    f for f in os.listdir(image_folder) 
    if os.path.isfile(os.path.join(image_folder, f)) and 
    Path(f).suffix.lower() in image_extensions
]

print(f"총 이미지 파일 수: {len(image_files)}")

# 50개보다 많을 경우에만 삭제 진행
if len(image_files) > 50:
    # 랜덤으로 50개 선택 (남길 파일들)
    files_to_keep = set(random.sample(image_files, 50))
    
    # 나머지 파일들 삭제
    files_to_delete = [f for f in image_files if f not in files_to_keep]
    
    for file in files_to_delete:
        file_path = os.path.join(image_folder, file)
        os.remove(file_path)
        print(f"삭제됨: {file}")
    
    print(f"\n삭제 완료! 남은 파일 수: {len(files_to_keep)}")
else:
    print(f"파일이 50개 이하입니다. 삭제하지 않습니다.")