from PIL import Image
import torchvision.transforms as transforms
import argparse
import torch
import os

CLASS = ['Landfill', 'Container/Recyclebles', 'Paper', 'Coffee Cups']
def main(args):
    transform = transforms.Compose([
                transforms.RandomResizedCrop(128, (1, 1), (1, 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    inception_transform = transforms.Compose([
                transforms.RandomResizedCrop(128, (1, 1), (1, 1)),
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = eval('{}transform'.format('inception_' if args.transfer else ''))
    model_path = eval('{}model.pt'.format('transfer_' if args.transfer else ''))
    model = torch.load(model_path)

    while True:
        x = input('Take a photo and press Enter to continue:\n')
        if x == 'quit':
            break
        if x != '':
            continue
        img = max([fname for fname in os.listdir(r'C:\Users\iatey\Pictures\Camera Roll') if fname.startswith('WIN')])
        img = transform(Image.open(r'C:\Users\iatey\Pictures\Camera Roll/'+img))
        print(img)
        prediction = model(img.unsqueeze())
        confidence, predicted = torch.max(prediction, 1)
        predicted = CLASS[predicted]
        description = input('Describe the item in less than four words:')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer', type=int, default=0)
    args = parser.parse_args()
    main(args)
