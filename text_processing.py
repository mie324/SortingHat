print('\nloading... please wait')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

wv = Word2Vec.load('./data/w2v_wiki300').wv

# garbage: 0, containers: 1, paper: 2, coffee cups: 3
KEYWORDS = {'snacks': 0, 'chips': 1, 'wrapper': 2, 'snacks packaging': 3,
            'juice box': 10,  'takeout box': 11, 'container': 12, 'bottle': 13,
            'can': 14, 'pop can': 15, 'glass bottle': 16, 'cutlery': 17,
            'newspaper': 20, 'paper': 21, 'paper tray': 22, 'napkins': 23,
            'coffee cup': 30, 'tim hortons': 31, 'starbucks': 32, 'coffee': 33
            }
CATEGORIES = ['landfill', 'containers', 'paper', 'coffee cups']
KEYWORDS_rev = {val:key for key, val in KEYWORDS.items()}
keywords_mat = np.zeros([len(KEYWORDS), 301]) # first dim is waste code, rest 300 vector
for i, (phrase, wc) in enumerate(KEYWORDS.items()):
    if ' ' in phrase:  # two words
        w1, w2 = phrase.split(' ')
        vec = wv[w1] + wv[w2]
    else:
        vec = wv[phrase]
    keywords_mat[i, 0] = wc
    keywords_mat[i, 1:] = vec

while True:
    my_phrase = input('Enter your phrase for simulated result: ').lower()
    if my_phrase == 'exit()':
        break
    words = my_phrase.split(' ')
    if len(words) > 4:
        print('try to keep your phrase more succinct.\n')
        continue
    my_vec= np.zeros(300)

    for w in words:
        try:
            my_vec += wv[w]
        except KeyError:
            print("I don't know what {} means, sorry. Did you make a typo?\n".format(w))
            break
    else: # no break
        #compute max similarity
        max_sim, max_lb = 0, -1
        for row in keywords_mat:
            sim = 1 - cosine(my_vec, row[1:])
            if sim > max_sim:
                max_sim = sim
                max_lb = row[0]

        print("--> I think '{}' is the closest to '{}' (similarity {:.4f}), so it should go in {}\n".format(
            my_phrase, KEYWORDS_rev[max_lb], max_sim, CATEGORIES[int(max_lb//10)]))