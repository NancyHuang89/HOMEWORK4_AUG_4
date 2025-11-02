import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_ego_cart: Boolean indicating if this is the ego car (track_id=0)
    """
    with open(info_path) as f:
        info = json.load(f)
    
    karts_list = info["karts"]
    detections = info["detections"]
    
    if view_index >= len(detections):
        return []
    
    frame_detections = detections[view_index]
    kart_objects = []
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Only process karts (class_id = 1)
        if class_id != 1:
            continue
        
        # Scale coordinates
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Calculate center
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Check if bounding box is valid and within image bounds
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Get kart name (track_id corresponds to index in karts list)
        # track_id=0 is ego car, track_id=1 is first other kart, etc.
        if track_id < len(karts_list):
            kart_name = karts_list[track_id]
        else:
            continue
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_ego_cart": (track_id == 0),
            "distance_down_track": info["distance_down_track"][track_id] if track_id < len(info["distance_down_track"]) else 0
        })
    
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    
    return info.get("track", "unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, image_file: str = None) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
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
    
    qa_pairs = []
    
    # 1. Ego car question: What kart is the ego car?
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_cart["kart_name"],
        "image_file": image_file
    })
    
    # 2. Total karts question: How many karts are there in the scenario?
    # Count all karts from the karts list
    total_karts = len(info["karts"])
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(total_karts),
        "image_file": image_file
    })
    
    # 3. Track information question: What track is this?
    track_name = extract_track_info(info_path)
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name,
        "image_file": image_file
    })
    
    # 4. Relative position questions for each kart (front/behind)
    # Use distance_down_track to determine front/behind
    ego_distance = ego_cart["distance_down_track"]
    ego_center_x, ego_center_y = ego_cart["center"]
    
    for kart in kart_objects:
        if kart["is_ego_cart"]:
            continue
        
        kart_distance = kart["distance_down_track"]
        kart_center_x, kart_center_y = kart["center"]
        
        # Determine front/behind based on distance_down_track
        # Higher distance = further down track = in front
        if kart_distance > ego_distance:
            position = "front"
        else:
            position = "behind"
        
        qa_pairs.append({
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": position,
            "image_file": image_file
        })
    
    # 5. Counting questions
    # Count karts in front
    karts_in_front = sum(1 for kart in kart_objects 
                        if not kart["is_ego_cart"] and kart["distance_down_track"] > ego_distance)
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(karts_in_front),
        "image_file": image_file
    })
    
    # Count karts behind
    karts_behind = sum(1 for kart in kart_objects 
                      if not kart["is_ego_cart"] and kart["distance_down_track"] < ego_distance)
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(karts_behind),
        "image_file": image_file
    })
    
    # Count karts to the left (x-coordinate smaller)
    karts_left = sum(1 for kart in kart_objects 
                    if not kart["is_ego_cart"] and kart["center"][0] < ego_center_x)
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(karts_left),
        "image_file": image_file
    })
    
    # Count karts to the right (x-coordinate larger)
    karts_right = sum(1 for kart in kart_objects 
                     if not kart["is_ego_cart"] and kart["center"][0] > ego_center_x)
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(karts_right),
        "image_file": image_file
    })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    image_file_rel = Path(image_file).relative_to(Path(info_file).parent.parent)
    qa_pairs = generate_qa_pairs(info_file, view_index, image_file=str(image_file_rel))

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        if 'image_file' in qa:
            print(f"Image: {qa['image_file']}")
        print("-" * 50)


def generate_all_qa_pairs(data_dir: str = "data/train", output_file: str = "data/train/train_qa_pairs.json"):
    """
    Generate QA pairs for all info.json files in the training directory.
    
    Args:
        data_dir: Directory containing info.json files and images
        output_file: Output JSON file path
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    # Find all info.json files
    info_files = list(data_path.glob("*_info.json"))
    print(f"Found {len(info_files)} info files")
    
    all_qa_pairs = []
    
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        
        # Try to find all image files for this info file
        image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))
        
        for image_file in image_files:
            # Extract view index from filename (format: BASENAME_VIEWINDEX_im.jpg)
            try:
                parts = image_file.stem.split("_")
                if len(parts) >= 2:
                    # The view index is the second-to-last part before "_im"
                    view_index = int(parts[-2])
                    
                    # Generate QA pairs for this view
                    # Make path relative to data directory (e.g., "train/00000_00_im.jpg")
                    image_file_rel = image_file.relative_to(data_path.parent)
                    qa_pairs = generate_qa_pairs(
                        str(info_file), 
                        view_index,
                        img_width=600,  # Original image width
                        img_height=400,  # Original image height
                        image_file=str(image_file_rel)
                    )
                    all_qa_pairs.extend(qa_pairs)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse view index from {image_file}: {e}")
                continue
        
        current_idx = list(info_files).index(info_file) + 1
        if current_idx % 100 == 0:
            print(f"Processed {current_idx}/{len(info_files)} files...")
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\nGenerated {len(all_qa_pairs)} QA pairs")
    print(f"Saved to {output_path}")


"""
Usage Examples:
   # Visualize QA pairs for a specific file and view:
   python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
   
   # Generate all QA pairs for training data:
   python -m homework.generate_qa generate_all --data_dir data/train --output_file data/train/train_qa_pairs.json
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all_qa_pairs})


if __name__ == "__main__":
    main()
