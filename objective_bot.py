from PIL import Image
import torchvision.transforms as transforms
import argparse
import torch
import os
import numpy as np
from text_processing import baseline

CLASS = ['Landfill', 'Container/Recyclebles', 'Paper', 'Coffee Cups']


def softmax(w):
    e = np.exp(np.array(w))
    return e / np.sum(e)


def comb_func(cnn_res, nlp_res):
    cnn_res = softmax(cnn_res)  # probably already softmax
    nlp_res = softmax(nlp_res)
    return (cnn_res + nlp_res)/2


def main(args):

    resolution = 299 if args.transfer else 128

    transform = transforms.Compose([
                transforms.RandomResizedCrop(resolution, (1, 1), (1, 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model_path = '{}model.pt'.format('transfer_' if args.transfer else '')
    model = torch.load(model_path)

    while True:
        ## CNN
        x = input('Take a photo and press Enter to continue:\n')
        if x == 'quit':
            break
        if x != '':
            continue
        img = max([n for n in os.listdir(r'C:\Users\iatey\Pictures\Camera Roll') if n.startswith('WIN')])
        img = transform(Image.open(r'C:\Users\iatey\Pictures\Camera Roll/'+img))
        print(img)
        prediction = model(img.unsqueeze(0))
        confidences = [ sum(prediction[2,3,7,8]), sum(prediction[0,4,5,6,10], prediction[1], prediction[9]) ]

        #confidence, predicted = torch.max(prediction, 1)
        pred_cnn = CLASS[np.argmax(confidences)]

        ## NLP
        words = input('Describe the item in less than four words:').split(' ')
        max_lbs, max_sims = baseline(words)

        pred_nlp = CLASS[max_lbs[np.argmax(max_sims)]//10]

        combined = comb_func(confidences, max_sims)
        if args.debug:
            print('     Landfill  Container  Paper  Coffee')
            print('CNN:  {}'.format(confidences))
            print('NLP:  {}'.format(max_sims))
            print('COMB: {}'.format(combined))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    main(args)
