import os
import pandas as pd

IMAGE_DIR = r"../images"
metadata = pd.read_csv("metadata.csv")
# missing_files = [img for img in metadata["image"] if not os.path.exists(os.path.join(IMAGE_DIR, img))]

# if missing_files:
#     print(f"Missing files: {missing_files}")
# else:
#     print("All files are present.")

# print(metadata.head())

metadata['image'] = metadata['image'].apply(lambda x: os.path.join(IMAGE_DIR, x))

print(metadata[['id', 'name', 'description', 'price', 'Qty', 'image' ]])