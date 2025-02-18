import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
data_dir = r"D:\Varsha\AIML"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Split Dataset into Train and Test
def split_dataset():
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in ["non-covid", "covid"]:
        category_path = os.path.join(data_dir, category)

        if not os.path.exists(category_path):
            print(f"Category folder not found: {category_path}")
            return

        images = os.listdir(category_path)
        if len(images) == 0:
            print(f"No images found in {category_path}")
            return

        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Create subfolders for each category in train and test directories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        for image in train_images:
            shutil.move(os.path.join(category_path, image), os.path.join(train_dir, category, image))
        for image in test_images:
            shutil.move(os.path.join(category_path, image), os.path.join(test_dir, category, image))

    print("Dataset splitting complete.")

split_dataset()
