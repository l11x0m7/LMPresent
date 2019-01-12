'''
ELMo usage example with character inputs.

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import numpy as np
import spacy
from tqdm import tqdm
import json

nlp = spacy.blank('en')
INF = 1e30

debug = False

# Location of pretrained LM.  Here we use the test fixtures.
datadir = 'model'
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

out_data = '../data/RACE/feature'
raw_data = '../data/RACE/RACE'

if not os.path.exists(out_data):
    os.path.mkdir(out_data)


def softmax_mask(inputs, mask):
    """ Mask the padding values which may affect the softmax calculation.
    inputs: any shape
    mask: the same shape as `inputs`

    """
    return -INF * (1. - tf.cast(mask, tf.float32)) + inputs



def bilinear_attention(document, query, document_mask):
    """
    # document: (B, D, 2h)
    # query: (B, 2h)
    # document_mask: (B, D)

    # return: (B, 2h)
    """
    num_units = int(document.shape[-1])
    with tf.variable_scope('bilinear_att') as vs:
        W_att = tf.get_variable('W_bilinear', shape=(num_units, num_units), 
                dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
        M = tf.expand_dims(tf.matmul(query, W_att), axis=1)
        alpha = tf.nn.softmax(softmax_mask(tf.reduce_sum(document * M, axis=2), document_mask))
        return tf.reduce_sum(document * tf.expand_dims(alpha, axis=2), axis=1)

def bilinear_dot(attention, options):
    """
    # attention: (B, 2h)
    # options: (B, O=4, 2h)

    # return: (B, O=4)
    """
    num_units = int(attention.shape[-1])
    with tf.variable_scope('bilinear_dot') as vs:
        W_bili = tf.get_variable('W_bilinear', shape=(num_units, num_units), 
                      dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
        M = tf.expand_dims(tf.matmul(attention, W_bili), axis=1)
        alpha = tf.nn.softmax(tf.reduce_sum(options * M, axis=2))
        return alpha

def word_tokenize(sent):
    doc = nlp(sent)
    return ' '.join([token.text for token in doc])

def gen_data():
    vocabs = set()
    vocabs.add("<UNK>")
    vocabs.add("<S>")
    vocabs.add("</S>")
    for data_set in ["train", "dev", "test"]:
        for d in ['middle', 'high']:
            with open(os.path.join(out_data, '_'.join([data_set, d]) + '.json'), 'w') as fw:
                save_res = []
                raw_dir = os.path.join(raw_data, data_set, d)
                for dir_file in tqdm(os.listdir(raw_dir)):
                    obj = json.load(open(os.path.join(raw_dir, dir_file), "r"))
                    article = obj["article"].replace("\\newline", "\n").strip()
                    article = word_tokenize(article)
                    for word in article.strip().split():
                        vocabs.add(word)
                    for i, question in enumerate(obj['questions']):
                        res = {}
                        res["article"] = article
                        res["question"] = word_tokenize(obj["questions"][i].strip())
                        for word in res["question"].strip().split():
                            vocabs.add(word)
                        res["options"] = obj["options"][i]
                        res["answer"] = ord(obj["answers"][i]) - ord("A")
                        for k in range(4):
                            res["options"][k] = word_tokenize(res["options"][k].strip())
                            for word in res["options"][k].strip().split():
                                vocabs.add(word)
                        save_res.append(res)
                json.dump(save_res, fw)

def load_data():
    data_sets = ["train", "dev", "test"]
    ds = ['middle', 'high']
    train_datas = []
    for d in ds:
        with open(os.path.join(out_data, '_'.join([data_sets[0], d]) + '.json')) as fr:
            X = json.load(fr)
            train_datas.extend(X)
    val_datas = []
    for d in ds:
        with open(os.path.join(out_data, '_'.join([data_sets[1], d]) + '.json')) as fr:
            X = json.load(fr)
            val_datas.extend(X)
    test_datas = []
    test_h_datas = []
    test_m_datas = []
    for d in ds:
        with open(os.path.join(out_data, '_'.join([data_sets[2], d]) + '.json')) as fr:
            X = json.load(fr)
            test_datas.extend(X)
            if d == "middle":
                test_m_datas = X
            else:
                test_h_datas = X
    print(len(train_datas), len(val_datas), len(test_datas), len(test_m_datas), len(test_h_datas))
    return train_datas, val_datas, test_datas, test_m_datas, test_h_datas


def transform_data(train_datas, all_tokens, batch_size):
    train_context = [s['article'].split() for s in train_datas]
    train_question = [s['question'].split() for s in train_datas]
    train_options = [[k.split() for k in s['options']] for s in train_datas]
    train_answer = [s['answer'] for s in train_datas]

    assert len(train_context) == len(train_question) == len(train_options) == len(train_answer)

    # Now we can compute embeddings.
    for sentence in train_context + train_question:
        for token in sentence:
            all_tokens.add(token)
    for options in train_options:
        for sentence in options:
            for token in sentence:
                all_tokens.add(token)
    

    train_context_length = [min(len(_), max_context_length - 2) for _ in train_context]
    train_question_length = [min(len(_), max_q_o_length - 2) for _ in train_question]
    train_options_length = [[min(len(__), max_q_o_length - 2) for __ in _] for _ in train_options]
    train_sort_index, _ = zip(*sorted(zip(range(len(train_context_length)), train_context_length), key=lambda k: k[1], reverse=False))
    train_sort_index = np.asarray(train_sort_index)
    train_context = np.asarray(train_context)[train_sort_index]
    train_question = np.asarray(train_question)[train_sort_index]
    train_options = np.asarray(train_options)[train_sort_index]
    train_answer = np.asarray(train_answer)[train_sort_index].astype(np.int32)

    train_context_length = np.asarray(train_context_length)[train_sort_index]
    train_question_length = np.asarray(train_question_length)[train_sort_index]
    train_options_length = np.asarray(train_options_length)[train_sort_index]

    train_sample_num = len(train_context)

    train_context = [train_context[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]
    train_question = [train_question[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]
    train_options = [train_options[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]
    train_answer = [train_answer[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]

    train_context_length = [train_context_length[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]
    train_question_length = [train_question_length[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]
    train_options_length = [train_options_length[k:min(k+batch_size, train_sample_num)] for k in range(0, train_sample_num, batch_size)]

    return (train_context, train_question, train_options, train_answer, train_context_length, train_question_length, train_options_length), train_sample_num


def evaluate(datas, batch_num=-1):
    if batch_num < 0:
        batch_num = len(datas[0])
    elif batch_num > len(datas[0]):
        batch_num = len(datas[0])
    total_loss = []
    total_preds = []
    total_labels = []
    if debug:
        batch_num = 1
    for _ in range(batch_num):
        batch_context, batch_question, batch_options, batch_answer = datas[0][_], datas[1][_], datas[2][_], datas[3][_]
        batch_context_l, batch_question_l, batch_options_l = datas[4][_], datas[5][_], datas[6][_]
        batch_options = batch_options.flatten()
        batch_options_l = batch_options_l.flatten()
        context_ids = batcher.batch_sentences(batch_context)
        question_ids = batcher2.batch_sentences(batch_question)
        options_ids = batcher2.batch_sentences(batch_options)
        out_loss, preds = sess.run(
                [losses, predictions],
                feed_dict={context_character_ids: context_ids,
                           question_character_ids: question_ids,
                           options_character_ids: options_ids,
                           context_lengths: batch_context_l, 
                           question_lengths: batch_question_l, 
                           options_lengths: batch_options_l, 
                           labels: batch_answer}
            )
        total_loss.extend(out_loss)
        total_preds.extend(preds)
        total_labels.extend(batch_answer)
    loss = np.mean(total_loss)
    accu = np.mean(np.asarray(total_preds) == np.asarray(total_labels))
    return loss, accu



max_context_length = 512
max_q_o_length = 32
all_tokens = set(['<S>', '</S>'])
batch_size = 8

# load data
try:
    train_datas, val_datas, test_datas, test_m_datas, test_h_datas = load_data()
except:
    gen_data()
    train_datas, val_datas, test_datas, test_m_datas, test_h_datas = load_data()

train_datas, train_sample_num = transform_data(train_datas, all_tokens, batch_size)
train_batch_num = len(train_datas[0])
val_datas, val_sample_num = transform_data(val_datas, all_tokens, batch_size)
val_batch_num = len(val_datas[0])
test_datas, test_sample_num = transform_data(test_datas, all_tokens, batch_size)
test_batch_num = len(test_datas[0])
test_m_datas, test_m_sample_num = transform_data(test_m_datas, all_tokens, batch_size)
test_m_batch_num = len(test_m_datas[0])
test_h_datas, test_h_sample_num = transform_data(test_h_datas, all_tokens, batch_size)
test_h_batch_num = len(test_h_datas[0])

# build and save vocab file
with open(vocab_file, 'w') as fout:
        fout.write('\n'.join(all_tokens))


# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50, max_context_length)
batcher2 = Batcher(vocab_file, 50, max_q_o_length)






# *** build models ***

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
question_character_ids = tf.placeholder('int32', shape=(None, None, 50))
options_character_ids = tf.placeholder('int32', shape=(None, None, 50))
context_lengths = tf.placeholder('int32', shape=(None, ))
question_lengths = tf.placeholder('int32', shape=(None, ))
options_lengths = tf.placeholder('int32', shape=(None, ))
labels = tf.placeholder('int32', shape=(None, ))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
question_embeddings_op = bilm(question_character_ids)
options_embeddings_op = bilm(options_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )
    elmo_options_input = weight_layers(
        'input', options_embeddings_op, l2_coef=0.0)

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )
    elmo_options_output = weight_layers(
        'output', options_embeddings_op, l2_coef=0.0)

with tf.variable_scope('last_layer'):
    context_mask = tf.sequence_mask(context_lengths)
    # [batch_size, q_len, dim]
    q_rep = tf.concat(
        [elmo_question_input['weighted_op'], elmo_question_output['weighted_op']], axis=2)
    # [batch_size, dim]
    q_rep = tf.reduce_mean(q_rep, axis=1)

    # [batch_size, c_len, dim]
    c_rep = tf.concat(
        [elmo_context_input['weighted_op'], elmo_context_output['weighted_op']], axis=2)

    # [batch_size, dim]
    c_q_rep = bilinear_attention(c_rep, q_rep, context_mask)

    o_rep = tf.concat(
        [elmo_options_input['weighted_op'], elmo_options_output['weighted_op']], axis=2)
    o_rep = tf.reduce_mean(o_rep, axis=1)
    o_rep = tf.reshape(o_rep, (-1, 4, int(o_rep.shape[-1])))
    out = bilinear_dot(c_q_rep, o_rep)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=out,
                        labels=labels
                    )
    loss = tf.reduce_mean(losses)
    predictions = tf.cast(tf.argmax(out, axis=1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(out, axis=1), tf.int32), labels), tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=8e-5, epsilon=1e-6).minimize(loss)



import time

t1 = time.time()
epoch_nums = 25

# train and evaluate model
with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    best_val_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(epoch_nums):
        inds = list(range(len(train_datas[0])))
        np.random.shuffle(inds)
        train_datas = [np.asarray(_)[inds] for _ in train_datas]
        for (i, (batch_context, batch_question, batch_options, batch_answer, \
            batch_context_l, batch_question_l, batch_options_l)) in enumerate(
            zip(*train_datas)):
            batch_options = batch_options.flatten()
            context_ids = batcher.batch_sentences(batch_context)
            question_ids = batcher2.batch_sentences(batch_question)
            options_ids = batcher2.batch_sentences(batch_options)
            max_context_length = context_ids.shape[1]
            batch_options_l = batch_options_l.flatten()

            # Compute ELMo representations (here for the input only, for simplicity).
            _, out_loss, accu = sess.run(
                [train_op, loss, accuracy],
                feed_dict={context_character_ids: context_ids,
                           question_character_ids: question_ids,
                           options_character_ids: options_ids,
                           context_lengths: batch_context_l, 
                           question_lengths: batch_question_l, 
                           options_lengths: batch_options_l, 
                           labels: batch_answer}
            )
            if i % 10 == 0 and i != 0 or debug:
                print("***epoch: {}, iter: {}/{}, time: {}***".format(epoch, i, train_batch_num, time.time() - t1))
                print(out_loss, accu, max_context_length)
            if debug:
                break
        tr_loss, tr_accu = evaluate(train_datas, val_batch_num)
        val_loss, val_accu = evaluate(val_datas)
        test_loss, test_accu = evaluate(test_datas)
        tm_loss, tm_accu = evaluate(test_m_datas)
        th_loss, th_accu = evaluate(test_h_datas)
        if best_val_acc < val_accu:
            best_val_acc = val_accu
            best_test_acc = test_accu
        print("[train]loss: {}, accu: {}".format(tr_loss, tr_accu))
        print("[val]loss: {}, accu: {}".format(val_loss, val_accu))
        print("[test]loss: {}, accu: {}".format(test_loss, test_accu))
        print("[test middle]loss: {}, accu: {}".format(tm_loss, tm_accu))
        print("[test high]loss: {}, accu: {}".format(th_loss, th_accu))
        print("[best val]accu: {}".format(best_val_acc))
        print("[test on best val]accu: {}".format(best_test_acc))

