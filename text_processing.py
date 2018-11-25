import warnings
warnings.filterwarnings("ignore")
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import pandas as pd

wv = Word2Vec.load('./data/w2v_wiki300').wv

# garbage: 0, containers: 1, paper: 2, coffee cups: 3
KEYWORDS = {'snacks': 0, 'chips': 1, 'wrapper': 2, 'snacks packaging': 3, 'food': 4, 'organic': 5, 'fruit': 6,
            'juicebox': 10,  'takeout box': 11, 'container': 12, 'bottle': 13,
            'can': 14, 'pop can': 15, 'glass bottle': 16, 'cutlery': 17, 'plastic cup': 18, 'tea cup': 19,
            'newspaper': 20, 'paper': 21, 'paper tray': 22, 'napkins': 23, 'magazine': 24, 'cup sleeve': 25,
            'coffee cup': 30, 'tim hortons': 31, 'starbucks': 32, 'coffee': 33, 'paper coffee cup': 34
            }
CATEGORIES = ['landfill', 'containers', 'paper', 'coffee cups']
KEYWORDS_rev = {val:key for key, val in KEYWORDS.items()}
keywords_mat = np.zeros([len(KEYWORDS), 301]) # first dim is waste code, rest 300 vector
for i, (phrase, wc) in enumerate(KEYWORDS.items()):
    if ' ' in phrase:  # two words
        words = phrase.split(' ')
        vec = sum([wv[w] for w in words])
    else:
        vec = wv[phrase]
    keywords_mat[i, 0] = wc
    keywords_mat[i, 1:] = vec

def baseline(words, return_all_class = False):
    ''' words: list of strings
    return max_lb and max_sim if return_all_class = False
    else, return max similarity of words in each of the four bins
    '''
    my_vec = np.zeros(300)
    for w in words:
        try:
            my_vec += wv[w]
        except KeyError:
            print("I don't know what {} means, sorry. Did you make a typo?\n".format(w))
            return None

    # compute max similarity
    max_sims, max_lbs = [0, 0, 0, 0], [-1, -1, -1, -1]
    for row in keywords_mat:
        bin = int(row[0]//10)
        sim = 1 - cosine(my_vec, row[1:])
        if sim > max_sims[bin]:
            max_sims[bin] = sim
            max_lbs[bin] = row[0]
    if return_all_class:
        return max_lbs, max_sims
    else:
        max_bin = max_sims.index(max(max_sims))
        return max_lbs[max_bin], max_sims[max_bin]


def bot():
    while True:
        my_phrase = input('Enter your phrase for simulated result: ').lower()
        if my_phrase == 'exit()':
            break
        words = my_phrase.replace('-', ' ').split(' ').lower()
        if len(words) > 4:
            print('try to keep your phrase more succinct.\n')
            continue

        ans = baseline(words)
        if ans is not None:
            max_lb, max_sim = ans
            print("--> I think '{}' is the closest to '{}' (similarity {:.4f}), so it should go in {}\n".format(
                my_phrase, KEYWORDS_rev[max_lb], max_sim, CATEGORIES[int(max_lb//10)]))

def test_bot():
    df = pd.read_csv('./data/waste_wizard.csv', names=['words', 'cat'])
    df = df[df['cat'] != 4]
    X = df['words'].values
    y = df['cat'].values
    corr = 0
    ans = []
    for i, (phrase, label) in enumerate(zip(X,y)):
        words = phrase.replace('-', ' ').split(' ')
        max_lb, _ = baseline(words)

        ans.append(max_lb//10)
        if max_lb//10 == label:
            corr += 1
    df['pred'] = ans
    df.to_csv('./data/waste_wizard_answers.csv', index=False)
    return corr, i+1

def train_on_wv():
    pass


if __name__ == '__main__':
    print('\nloading... please wait')
    #print(test_bot())
    bot()