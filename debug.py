from PIL import Image
import torchvision.transforms as transforms
import os

CLASSES_itos = ['plasticbottle', 'newspaper', 'plasticbag', 'perishable',
            'glassbottle', 'popcan', 'juicebox', 'ffwrapper',
            'snackpackage', 'coffeecups', 'togobox']


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for clsname in ['paper']:
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





    '''
    Todo:
    NLP:
        run baseline nlp model with new set of keywords and get a numerical performance
        write and train nn for nlp and get an accuracy
    CNN:
        test model performance on test set
        try transfer learning and see if accuracy increases
        plot training and validation curves
        
        
    Integration:
        combine cnn and nlp and play with weighting
        test combined model with test set
        
        think about how to demo??
        >>> pic: 'xxx.png', phrase = 'coffee cup'
        >>> CNN: containers ()
            NLP: coffee cup ()
            Combined: coffee cup ()
    
    
    
    Today:
        
    '''