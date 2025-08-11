from icrawler.builtin import GoogleImageCrawler
import os
import glob
import cv2
import shutil
import random
import re

# Step 1: Generate Queries
def generate_search_queries():
    expressions = ["happy", "sad", "angry", "neutral", "surprised", "disgusted"]
    descriptors = [
        "Black woman", "Black man", "Black boy", "Black girl", "Black toddler", "Black elderly man", "Black elderly woman",
        "White woman", "White man", "White boy", "White girl", "White toddler", "White elderly man", "White elderly woman",
        "Middle Eastern woman", "Middle Eastern man", "Middle Eastern boy", "Middle Eastern girl", "Middle Eastern toddler", "Middle Eastern elderly man", "Middle Eastern elderly woman",
        "Asian woman", "Asian man", "Asian toddler","Asian girl", "Asian elderly man", "Asian elderly woman",
        "Indian woman", "Indian man", "Indian boy", "Indian girl", "Indian toddler", "Indian elderly man", "Indian elderly woman",
        "mixed race people",
    ]

    queries = []
    for expr in expressions:
        for desc in descriptors:
            q = f"{expr} {desc} face photo"
            queries.append(q)
    return queries

# Step 2: Collect Images
def collect_images_by_keywords(base_dir="faces", keyword_list=[], max_images=500):
    for keyword in keyword_list:
        folder_name = keyword.replace(" ", "_").replace(",", "").lower()
        output_path = os.path.join(base_dir, folder_name)
        os.makedirs(output_path, exist_ok=True)

        print(f"Downloading: {keyword}")
        crawler = GoogleImageCrawler(storage={'root_dir': output_path})
        crawler.crawl(keyword=keyword, max_num=max_images, filters={'size': 'medium', 'type': 'photo'})

# Step 3: Remove Known Bad Folders
def delete_folder(folder_path):
    shutil.rmtree(folder_path, ignore_errors=True)
    print(f"Removed: {folder_path}")

def remove_known_bad_folders(base_dir="faces", folder_names=[]):
    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        delete_folder(folder_path)

# Step 4: Remove Images Without Faces
def clean_images_without_faces(base_dir="faces"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cleaned = 0
    removed = 0

    print("\n Cleaning images without faces...")
    for path in glob.glob(f"{base_dir}/**/*.jpg", recursive=True):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print(f"No face: {path}")
            os.remove(path)
            removed += 1
        else:
            cleaned += 1

    print(f"Cleanup complete: {cleaned} kept, {removed} removed.\n")

# Step 5: Add Blurred Face Variants
def blur_random_faces(base_dir="faces", blur_ratio=0.1):
    print(f"Blurring ~{int(blur_ratio*100)}% of images...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    all_images = glob.glob(f"{base_dir}/**/*.jpg", recursive=True)
    selected_images = random.sample(all_images, int(len(all_images) * blur_ratio))

    for path in selected_images:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_region = img[y:y+h, x:x+w]
                face_region = cv2.GaussianBlur(face_region, (31, 31), 30)
                img[y:y+h, x:x+w] = face_region
        else:
            img = cv2.GaussianBlur(img, (15, 15), 15)

        new_path = path.replace(".jpg", "_blurred.jpg")
        cv2.imwrite(new_path, img)

    print(f" Blurred {len(selected_images)} images.\n")

# Step 6: Organize for ML Training
def organize_for_ml(source_dir="faces", output_dir="dataset"):
    print(f"\n Organizing into '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    for folder in folders:
        match = re.match(r"^(happy|sad|angry|neutral|surprised|disgusted)", folder)
        if not match:
            print(f"Skipping unknown: {folder}")
            continue

        label = match.group(1)
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        image_paths = glob.glob(os.path.join(source_dir, folder, "*.jpg"))
        for img_path in image_paths:
            fname = os.path.basename(img_path)
            dest = os.path.join(label_dir, f"{folder}_{fname}")
            shutil.copyfile(img_path, dest)

    print(" Dataset organized!\n")

# Step 7: Split dataset into train/val/test
def split_dataset_for_ml(input_dir="dataset", output_dir="final_dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print(f"\nSplitting dataset into train/val/test...")
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    splits = ["train", "val", "test"]
    for split in splits:
        for label in os.listdir(input_dir):
            src_label_dir = os.path.join(input_dir, label)
            if not os.path.isdir(src_label_dir):
                continue
            dst_dir = os.path.join(output_dir, split, label)
            os.makedirs(dst_dir, exist_ok=True)

    for label in os.listdir(input_dir):
        src_label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(src_label_dir):
            continue

        images = glob.glob(os.path.join(src_label_dir, "*.jpg"))
        random.shuffle(images)
        total = len(images)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        split_lists = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_images in split_lists.items():
            dst_dir = os.path.join(output_dir, split, label)
            for img_path in split_images:
                fname = os.path.basename(img_path)
                shutil.copyfile(img_path, os.path.join(dst_dir, fname))

    print("Dataset split complete!\n")

# # Pipeline Runner
# if __name__ == "__main__":
#     base_dir = "faces"
#     queries = generate_search_queries()

#     # Step 1: Download
#     collect_images_by_keywords(base_dir=base_dir, keyword_list=queries, max_images=10)

#     # Step 2: Clean known bad folders
#     remove_known_bad_folders(base_dir=base_dir, folder_names=["blurry_happy_face_picture"])

#     # Step 3: Remove images without faces
#     clean_images_without_faces(base_dir=base_dir)

#     # # Step 4: Add blurred variants
#     #blur_random_faces(base_dir=base_dir, blur_ratio=0.1)

#     # Step 5: Organize for ML training
#     organize_for_ml(source_dir=base_dir, output_dir="dataset")

#     # Step 6: Split into train/val/test
#     split_dataset_for_ml(input_dir="dataset", output_dir="final_dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

#     print("All done!")"""## Save into drive"""

# # copy images to my drive
import shutil

# # Create a directory in your Google Drive
!mkdir -p /content/drive/MyDrive

# # Copy the  folder
# shutil.copytree("final_dataset", "/content/drive/MyDrive/face_data", dirs_exist_ok=True)

!cp -r /content/dataset /content/drive/MyDrive/face_data/facial_dataset

!ls /content/drive/MyDrive/facial_dataset/

