from torchvision import transforms
from PIL import Image
import glob
import os
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# 40x: togobox, coffeecups, ffwraper, juicebox, snackpackage

Jitter = transforms.ColorJitter(0.1, 0.4, 0.4)
Affine = transforms.RandomAffine(180, (0.05, 0.05), (0.9, 1.1))
Std_crop = transforms.RandomResizedCrop(128, (1, 1), (1, 1))
Crop = transforms.RandomResizedCrop(128, (0.5, 1))

errcount = 0

#for fd in ['togobox', 'coffeecups', 'ffwrapper', 'juicebox', 'snackpackage']:
#for fd in ['popcan', 'glassbottle', 'perishable']:
for fd in ['plasticbag', 'newspaper', 'plasticbottle']:
    os.mkdir('./data/{}/'.format(fd))

    imgcount = 0
    logging.info('Currently at class {}'.format(fd))
    #image_list = []
    for fmt in ['jpg', 'jpeg', 'png']:
        logging.info('-- at format {}'.format(fmt))

        for filename in glob.glob('../ImageNet_Utils/{}/*.{}'.format(fd, fmt)):

            if imgcount % 100 == 1:
                logging.info('--at {}st image'.format(imgcount))

            im = Image.open(filename)

            #image_list.append(Std_crop(im))
            try:
                for i in range(2):
                    imafterjit = Jitter(im)
                    for j in range(4):
                        imafteraf = Affine(imafterjit)
                        for k in range(2):
                            img = Crop(imafteraf)
                            imgcount += 1
                            img.save('./data/{}/{}{}.png'.format(fd, fd, imgcount))

            except ValueError as e:
                errcount += 1
                print(errcount, e)
    # for i, img in enumerate(image_list):
    #     img.save('./data/{}/{}{}.png'.format(fd, fd, i))




