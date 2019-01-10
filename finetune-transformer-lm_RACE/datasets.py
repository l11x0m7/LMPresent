import os
import csv
import numpy as np

from tqdm import tqdm
import tensorflow as tf
import json

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path) as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)


def get_file(path):
    files = []
    for inf in os.listdir(path):
        new_path = os.path.join(path, inf)
        if os.path.isdir(new_path):
            assert inf in ["middle", "high"]
            files += get_file(new_path)
        else: 
            if new_path.find(".DS_Store") != -1:
                continue
            files += [new_path]
    return files

def race(data_dir, is_train=True):
    def get_one_dir_data(data_dir):
        article_questions = []
        opts = [[], [], [], []]
        ys = []
        unique_id = 0
        unique_id_for_opts = 0
        files = get_file(data_dir)
        count = 0
        for (i, fpath) in enumerate(files):
            with tf.gfile.GFile(fpath, "r") as reader:
                obj = json.load(reader)
                obj["article"] = obj["article"].replace("\\newline", "\n")
                for i in range(len(obj["questions"])):
                    ans = ord(obj["answers"][i]) - ord('A')
                    ys.append(ans)
                    article_questions.append(str(obj["article"] + " " + obj["questions"][i]))
                    for k in range(4):
                        opts[k].append(str(obj["options"][i][k]))
                        unique_id_for_opts += 1
                    unique_id += 1
            count += 1
            # if count == 100:
                # break
        return article_questions, opts[0], opts[1], opts[2], opts[3], ys
    if is_train:
        trX1, trX2, trX3, trX4, trX5, trY = get_one_dir_data(os.path.join(data_dir, 'train'))
        vaX1, valX2, vaX3, vaX4, vaX5, vaY = get_one_dir_data(os.path.join(data_dir, 'dev'))
        teX1, teX2, teX3, teX4, teX5, teY = get_one_dir_data(os.path.join(data_dir, 'test'))
        trY = np.asarray(trY, dtype=np.int32)
        vaY = np.asarray(vaY, dtype=np.int32)
        teY = np.asarray(teY, dtype=np.int32)
        return (trX1, trX2, trX3, trX4, trX5, trY), (vaX1, valX2, vaX3, vaX4, vaX5, vaY), (teX1, teX2, teX3, teX4, teX5, teY)
    else:
        vaX1, valX2, vaX3, vaX4, vaX5, vaY = get_one_dir_data(os.path.join(data_dir, 'dev'))
        teX1, teX2, teX3, teX4, teX5, teY = get_one_dir_data(os.path.join(data_dir, 'test'))
        m_teX1, m_teX2, m_teX3, m_teX4, m_teX5, m_teY = get_one_dir_data(os.path.join(data_dir, 'test', 'middle'))
        h_teX1, h_teX2, h_teX3, h_teX4, h_teX5, h_teY = get_one_dir_data(os.path.join(data_dir, 'test', 'high'))
        return (vaX1, valX2, vaX3, vaX4, vaX5, vaY), (teX1, teX2, teX3, teX4, teX5, teY), \
               (m_teX1, m_teX2, m_teX3, m_teX4, m_teX5, m_teY), (h_teX1, h_teX2, h_teX3, h_teX4, h_teX5, h_teY)

    

