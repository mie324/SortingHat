from torchvision import transforms
from PIL import Image
import torch
import glob
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

transform = transforms.Compose([
                transforms.RandomResizedCrop(128, (1, 1), (1, 1)),
                transforms.Resize(299), # 299 for inception / 224 for resnet
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_list = []
labels = []
CLASSES = ['landfill', 'container', 'paper', 'coffeecups']
for label in range(4):
    fd = CLASSES[label]
    logging.info('Currently at class {}'.format(fd))
    for fmt in ['png']:
        logging.info('-- at format {}'.format(fmt))
        for filename in glob.glob('./data/{}/*.{}'.format(fd, fmt)):
                im = Image.open(filename)
                im = transform(im)
                labels.append(label)
                img_list.append(im)
img_tensor = torch.Tensor(img_list)