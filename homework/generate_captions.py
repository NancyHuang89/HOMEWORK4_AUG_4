import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def generate_caption(info_path: str, view_index: int, img_width: int = 600, img_height: int = 400) -> list:
    """
    Generate captions for a specific view.
    
    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 600)
        img_height: Height of the image (default: 400)
    
    Returns:
        List of caption strings
    """
    with open(info_path) as f:
        info = json.load(f)
    
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    if not kart_objects:
        return []
    
    # Find ego car
    ego_cart = None
    for kart in kart_objects:
        if kart["is_ego_cart"]:
            ego_cart = kart
            break
    
    if not ego_cart:
        return []
    
    captions = []
    
    # 1. Ego car caption: {kart_name} is the ego car.
    captions.append(f"{ego_cart['kart_name']} is the ego car.")
    
    # 2. Counting caption: There are {num_karts} karts in the scenario.
    total_karts = len(info["karts"])
    captions.append(f"There are {total_karts} karts in the scenario.")
    
    # 3. Track name caption: The track is {track_name}.
    track_name = extract_track_info(info_path)
    captions.append(f"The track is {track_name}.")
    
    # 4. Relative position captions: {kart_name} is {position} of the ego car.
    ego_distance = ego_cart["distance_down_track"]
    
    for kart in kart_objects:
        if kart["is_ego_cart"]:
            continue
        
        kart_distance = kart["distance_down_track"]
        
        # Determine front/behind based on distance_down_track
        if kart_distance > ego_distance:
            position = "in front"
        else:
            position = "behind"
        
        captions.append(f"{kart['kart_name']} is {position} of the ego car.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaptions:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(data_dir: str = "data/train", output_file: str = "data/train/train_captions.json"):
    """
    Generate caption pairs for all info.json files in the training directory.
    
    Args:
        data_dir: Directory containing info.json files and images
        output_file: Output JSON file path
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    # Find all info.json files
    info_files = list(data_path.glob("*_info.json"))
    print(f"Found {len(info_files)} info files")
    
    all_captions = []
    
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        
        # Try to find all image files for this info file
        image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))
        
        for image_file in image_files:
            # Extract view index from filename (format: BASENAME_VIEWINDEX_im.jpg)
            try:
                parts = image_file.stem.split("_")
                if len(parts) >= 2:
                    view_index = int(parts[-2])
                    
                    # Generate captions for this view
                    captions = generate_caption(
                        str(info_file),
                        view_index,
                        img_width=600,
                        img_height=400
                    )
                    
                    # Create caption pairs (each caption gets its own entry)
                    image_file_rel = image_file.relative_to(data_path.parent)
                    for caption in captions:
                        all_captions.append({
                            "image_file": str(image_file_rel),
                            "caption": caption
                        })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse view index from {image_file}: {e}")
                continue
        
        current_idx = list(info_files).index(info_file) + 1
        if current_idx % 100 == 0:
            print(f"Processed {current_idx}/{len(info_files)} files...")
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"\nGenerated {len(all_captions)} caption pairs")
    print(f"Saved to {output_path}")


"""
Usage Examples:
   # Visualize captions for a specific file and view:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
   
   # Generate all caption pairs for training data:
   python -m homework.generate_captions generate_all --data_dir data/train --output_file data/train/train_captions.json
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all_captions})


if __name__ == "__main__":
    main()
