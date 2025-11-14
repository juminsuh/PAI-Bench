from icrawler.builtin import BingImageCrawler
import os

keywords = ["Eunwoo Cha", "Eunwoo Cha photoshoot", "Eunwoo Cha instagram", "Eunwoo Cha photo", "Eunwoo Cha face"]
celeb = "EunwooCha"
save_dir = f"./assets/images/{celeb}"
os.makedirs(save_dir, exist_ok=True)

for kw in keywords:
    crawler = BingImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=kw, max_num=300)
    
    