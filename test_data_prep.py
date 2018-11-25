from torchvision import transforms
from PIL import Image
import torch
import re
import glob
import logging
import numpy as np
from text_processing import baseline
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

resolution = 128  # 128 for Ours / 299 for Inception / 224 for ResNet
transform = transforms.Compose([
                transforms.RandomResizedCrop(resolution, (1, 1), (1, 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_test_data():
    img_list = []
    labels = []
    descriptions = []
    CLASSES = ['landfill', 'container', 'paper', 'coffeecups']
    for label in range(4):
        fd = CLASSES[label]
        logging.info('Currently at class {}'.format(fd))
        for fmt in ['jpg']:
            for filename in glob.glob('./test_data/{}/*.{}'.format(fd, fmt)):
                fname = re.sub(r'\./test_data/\w*\\', '', filename)
                descriptions.append(re.sub(r'.(JPG|jpg)$', '', fname))
                im = Image.open(filename)
                im = transform(im)
                labels.append(label)
                img_list.append(im)
    # img_tensor = torch.Tensor(img_list)
    return img_list, labels, descriptions


def softmax(w):
    e = np.exp(np.array(w))
    return e / np.sum(e)


def comb_func(cnn_res, nlp_res, alpha=0.5):
    ''' alpha is between 0 (pure nlp) and 1 (pure cnn) '''
    cnn_res = softmax(cnn_res)  # probably already softmax
    nlp_res = softmax(nlp_res)
    return cnn_res*alpha + nlp_res*(1-alpha)


def run_test():
    model = torch.load('model_1124_2104.pt').to('cpu').eval()

    imgs, labels, descrips = get_test_data()



    for alpha in np.linspace(0.1,0.9,9):
        corr = 0
        for i, (img, lb, des) in enumerate(zip(imgs, labels, descrips)):
            prediction = model(img.unsqueeze(0)).tolist() # for 4 cls

            ## NLP
            words = des.split(' ')
            max_lbs, max_sims = baseline(words)

            #pred_nlp = CLASS[max_lbs[np.argmax(max_sims)] // 10]

            combined = comb_func(prediction, max_sims, alpha = alpha)

            combined_pred = np.argmax(combined)

            corr += int(combined_pred == lb)



        logging.info('alpha = {:0.1f}: combined accuracy = {}'.format(alpha, corr/(i+1)))

if __name__ == '__main__':
    run_test()