from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
from test_data_prep import *
import pandas as pd
CLASS = ['Landfill', 'Container', 'Paper', 'Coffee']


def main(args):

    resolution = 299 if args.transfer else 128

    transform = transforms.Compose([
                transforms.RandomResizedCrop(resolution, (1, 1), (1, 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model_path = '{}model.pt'.format('transfer_' if args.transfer else '')
    model = torch.load(model_path, map_location='cpu').eval()

    while True:
        ## CNN
        x = input('Take a photo and press Enter to continue:\n')
        if x == 'quit':
            break
        if x != '':
            continue
        img = max([n for n in os.listdir(r'C:\Users\iatey\Pictures\Camera Roll') if n.startswith('WIN')])
        img = transform(Image.open(r'C:\Users\iatey\Pictures\Camera Roll/'+img))
        # print(img)
        prediction = softmax(model(img.unsqueeze(0)).squeeze().tolist())
        #confidences = [ sum(prediction[2,3,7,8]), sum(prediction[0,4,5,6,10], prediction[1], prediction[9]) ] # for 11 cls
        confidences = prediction  # for 4 cls

        #confidence, predicted = torch.max(prediction, 1)
        pred_cnn = CLASS[np.argmax(confidences)]

        ## NLP
        while True:
            words = input('Describe the item in less than four words: ').replace('-', ' ').lower().split(' ')
            ans = baseline(words)
            if ans is not None:
                break

        max_lbs, max_sims = ans
        max_sim_softmax = softmax(max_sims)
        pred_nlp = CLASS[ int( max_lbs[int(np.argmax(max_sims))]//10 ) ]

        combined = comb_func(confidences, max_sims)
        pred_comb = CLASS[np.argmax(combined)]

        df = pd.DataFrame(index=['CNN', 'NLP', 'COMB'], columns=['Pred', 'Landfill', 'Container', 'Paper', 'Coffee'])
        # df['Pred'] = [pred_cnn, pred_nlp, pred_com_softmax]
        df.loc['CNN', :] = [pred_cnn] + [round(e, 3) for e in confidences]
        df.loc['NLP', :] = [pred_nlp] + [round(e, 3) for e in max_sims]
        df.loc['COMB', :] = [pred_comb] + [round(e, 3) for e in combined]

        print(df)
        # print('         Landfill  Container  Paper  Coffee')
        # print('CNN:  {} | {}'.format(pred_cnn, confidences))
        # print('NLP:  {} | {}'.format(pred_nlp, max_sims))
        # print('COMB: {} | {}'.format(pred_comb, combined))
        print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    main(args)
