import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
from utility_functions import load_data, build_model, train_model, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument("data_directory", help="Path to the data directory")
    parser.add_argument("--save_dir", help="Directory to save checkpoints", default="checkpoints")
    parser.add_argument("--arch", help="Choose architecture", default="resnet50")
    parser.add_argument("--learning_rate", type=float, help="Set learning rate", default=0.001)
    parser.add_argument("--hidden_units", type=int, help="Set number of hidden units", default=512)
    parser.add_argument("--epochs", type=int, help="Set number of epochs", default=20)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Load and preprocess data
    trainloader, validloader, testloader, class_to_idx = load_data(args.data_directory)

    # Build the model
    model = build_model(args.arch, args.hidden_units, class_to_idx)

    # Train the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, args.gpu)

    # Save the checkpoint
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, args.learning_rate, class_to_idx)

if __name__ == "__main__":
    main()
