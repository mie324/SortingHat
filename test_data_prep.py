from torchvision import transforms
from PIL import Image
import torch
import re
import glob
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

transform = transforms.Compose([
                transforms.RandomResizedCrop(128, (1, 1), (1, 1)),
                transforms.Resize(299),  # 299 for Inception / 224 for ResNet
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def test():
    img_list = []
    labels = []
    descriptions = []
    CLASSES = ['landfill', 'container', 'paper', 'coffeecups']
    for label in range(4):
        fd = CLASSES[label]
        logging.info('Currently at class {}'.format(fd))
        for fmt in ['jpg']:
            for filename in glob.glob('./test/{}/*.{}'.format(fd, fmt)):
                fname = re.sub(r'\./test/\w*\\', '', filename)
                descriptions.append(re.sub(r'.JPG$', '', fname))
                im = Image.open(filename)
                im = transform(im)
                labels.append(label)
                img_list.append(im)
    # img_tensor = torch.Tensor(img_list)
    return img_list, labels, descriptions
