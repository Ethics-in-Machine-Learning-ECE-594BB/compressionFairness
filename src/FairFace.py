""" Instructions for setting up FairFace data locally:
1. git clone https://huggingface.co/datasets/HuggingFaceM4/FairFace
2. In fairface directory git lfs install
3. git lfs pull 
4. Run the following code
import pandas as pd
from PIL import Image
import io
df = pd.read_parquet("your_path.../FairFace/0.25/train-00000-of-00002-d405faba4f4b9b85.parquet")
for i in range(9000):
    im = df.iloc[i]['image']
    image = Image.open(io.BytesIO(im['bytes']))
    image.save(f"your_path.../CompressionFairness/data/images/train/faces/{im['path']}")
for i in range(9001,15000):
    im = df.iloc[i]['image']
    image = Image.open(io.BytesIO(im['bytes']))
    image.save(f"your_path.../CompressionFairness/data/images/test/faces/{im['path']}")    
5. Install: https://www.kaggle.com/datasets/jessicali9530/caltech256
6. Run the following code: 
import os 
import random
from math import floor
from tqdm import tqdm
source = 'your_path.../256_ObjectCategories/256_ObjectCategories/'
destination = 'your_path.../compressionFairness/data/images/train/non-faces'
allimages = os.listdir(source)
for dir in tqdm(allimages):
    curr_dir = os.path.join(source, dir)
    new_dir = os.listdir(curr_dir)
    index = floor(len(new_dir)*0.6)
    for img in new_dir[:index]:
        src_path = os.path.join(curr_dir, img)
        dst_path = os.path.join(destination, img)
        os.rename(src_path, dst_path)
destination = 'your_path.../compressionFairness/data/images/test/non-faces'
for dir in tqdm(allimages):
    curr_dir = os.path.join(source, dir)
    new_dir = os.listdir(curr_dir)
    index = floor(len(new_dir)*0.6)
    for img in new_dir[index:]:
        src_path = os.path.join(curr_dir, img)
        dst_path = os.path.join(destination, img)
        os.rename(src_path, dst_path)
Might be a more efficient way of doing this but idk
"""


"""
Rebalancing FairFace to be more reflective of other face datasets 

"""
import pandas as pd 
# TODO: evaluation loader for working with AIF360
