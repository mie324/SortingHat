from PIL import Image
import torchvision.transforms as transforms
import os

CLASSES_itos = ['plasticbottle', 'newspaper', 'plasticbag', 'perishable',
            'glassbottle', 'popcan', 'juicebox', 'ffwrapper',
            'snackpackage', 'coffeecups', 'togobox']


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for clsname in CLASSES_itos:
    c = 0
    numpics = len(os.listdir('./data/{}'.format(clsname)))

    for fnum in range(1,numpics+1):
        filename = './data/{}/{}{}.png'.format(clsname, clsname, fnum)
        try:
            im = Image.open(filename)
        except FileNotFoundError:
            continue
        imtensor = transform(im)
        if imtensor.shape != (3,128,128):
            #print(fnum, end=', ')
            os.rename(filename, './data/{}/{}{}_bad.png'.format(clsname, clsname, fnum))
            c += 1
    print(f'{clsname}: {c}/{numpics}')