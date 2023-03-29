import argparse
import csv
import helper
from concurrent.futures import ThreadPoolExecutor


def save_to_cache(image_data, csv_file):
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=image_data.keys())
        writer.writerow(image_data)


def check_cache(image_id, csv_file):
    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["image_id"] == image_id:
                    return True
    except FileNotFoundError:
        return False

    return False


def process_image(image_id, csv_file, output_dir):
    if check_cache(image_id, csv_file):
        print(f"Image {image_id} is already in cache, skipping.")
        return

    image = helper.download_image(image_id)
    basic_caption = helper.generate_basic_caption(image)
    expert_labeling_question = helper.generate_expert_labeling_question(basic_caption)
    expert_labels_critique = helper.generate_expert_labels_critique(image, expert_labeling_question)
    expert_caption = helper.convert_expert_conversation_data_to_labels(expert_labels_critique)

    image_data = {
        "image_id": image_id,
        "expert_caption": expert_caption,
    }

    image.save(expert_caption)

    save_to_cache(image_data, csv_file)
    return image_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using BLIP 2.0 and GPT-3.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process.")
    parser.add_argument("--image_dir", type=str, default=None, help="Image Dir")
    args = parser.parse_args()

    if args.image_dir is None:
        raise Error("no image dir")
    output_dir = args.image_dir

    csv_file = "processed_images_cache.csv"
    image_ids = helper.get_image_ids(args.num_images)


    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_image, image_ids, [csv_file] * len(image_ids), output_dir))

    print("Processed images:")
    for result in results:
        if result:
            print(f"Image ID: {result['image_id']}, Effective Labels: {result['effective_labels']}")
