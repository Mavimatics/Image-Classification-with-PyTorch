import argparse
from torchvision import models
from utility_functions import load_checkpoint, process_image, predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, help="Return top K most likely classes", default=1)
    parser.add_argument("--category_names", help="Use a mapping of categories to real names", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Preprocess the image
    img_tensor = process_image(args.image_path)

    # Predict the class
    probs, classes = predict(img_tensor, model, args.top_k, args.gpu)

    print("Probabilities:", probs)
    print("Classes:", classes)

if __name__ == "__main__":
    main()
