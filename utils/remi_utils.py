import re
import base64

def create_remi_dataset(split = "test"):
    from datasets import load_dataset
    from collections import defaultdict
    import json
    from pathlib import Path
    dataset = load_dataset("mehrankazemi/ReMI", split=split)
    train_dataset = defaultdict(dict)
    val_dataset = defaultdict(dict)
    for idx, data in enumerate(dataset):
        
        task = data['task']
        question = data['question']
        label = data['label']
        per_task_idx = len(train_dataset[task]) + len(val_dataset[task])
        if per_task_idx > 100:
            continue
        image_map = {f"<image{i}>": data[f'image_{i}'].convert("RGB") for i in range(1, 7) if data[f'image_{i}'] is not None}
        for image_map_key, image_map_value in image_map.items():
            Path(f"./data/REMI/{task}").mkdir(parents=True, exist_ok=True)
            save_as = f"./data/REMI/{task}/{idx}_{image_map_key}.png"
            image_map_value.save(save_as)
            image_map[image_map_key] = save_as
        
        problem_id = f"{data['task']}_{per_task_idx}"
        split = "train" if per_task_idx < 80 else "val"
        if split == "train":
            train_dataset[task][problem_id] = {
                "question": question,
                "label": label,
                "image_map": image_map,
                
            } 
        else:
            val_dataset[task][problem_id] = {
                "question": question,
                "label": label,
                "image_map": image_map,
            }
    
    for task in train_dataset.keys():
        with open(f"./data/REMI/train_{task}_dataset.json", "w") as f:
            json.dump(train_dataset[task], f, indent=4)
    for task in val_dataset.keys():
        with open(f"./data/REMI/val_{task}_dataset.json", "w") as f:
            json.dump(val_dataset[task], f, indent=4)

def load_remi(split = "val"):
    assert split in ["train", "val"]
    import json
    TASKS = ["FuncRead", "RefCoco", "Maps", "Collisions", "GeomCost", "Isomorphism", "IQ", "Charts", "EmojiAlgebra", "GeomShape", "CodeEdit", "Clocks"]
    all_dataset = {}
    for task in TASKS:
        with open(f"./data/REMI/{split}_{task}_dataset.json", "r") as f:
            dataset = json.load(f)
        all_dataset[task] = dataset
    return all_dataset

def load_remi_sample(sample, use_base64=False):
    from PIL import Image
    image_map = sample['image_map']
    prompt_splits = re.split(r'(<image\d+>)', sample['question'])
    
    content = []
    for splited in prompt_splits:
        splited = splited.strip()
        if not splited:
            continue
        if splited in sample['image_map']:
            if use_base64:
                image_path = image_map[splited]
                with open(image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                encoded_image_text = encoded_image.decode("utf-8")
                base64_qwen = f"data:image;base64,{encoded_image_text}"

                content.append({
                    "type": "image_url",
                    "image_url":{
                        "url": base64_qwen
                    }
                })
            else:
                content.append({
                    "type": "image",
                    "image": Image.open(image_map[splited]).convert("RGB")
                })
        else:
            content.append({
                "type": "text",
                "text": splited
            })
    return content, sample


def visualize_remi_sample(sample):
    import matplotlib.pyplot as plt 
    imgs = []
    for idx in range(1, 7): 
        image = sample['image_' + str(idx)]
        if image:
            imgs.append(image)
    # Display the images
    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    for img_idx, image in enumerate(imgs):
        axes[img_idx].imshow(image)
        axes[img_idx].axis('off')
        axes[img_idx].set_title(f'Image {img_idx + 1}')
        axes[img_idx].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
