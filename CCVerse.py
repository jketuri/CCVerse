#!/usr/bin/env python
# coding: utf-8

# In[43]:


from argparse import Namespace
from collections import Counter
import os
import re
from typing import Dict, List

import numpy as np

import pronouncing
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.word_to_vector import FastText

flags = Namespace(
    train_folder='data',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['day'],
    predict_top_k=5,
    checkpoint_path='checkpoint',
    n_epochs=200,
    verse_length=100,
    train_print=100,
    predict_print=100,
    line_length=8,
    stanza_length=4
)


def get_data_from_file(train_folder, batch_size, seq_size, words, topic_words):
    text = ''
    train_files = os.listdir(train_folder)
    for train_file in train_files:
        with open(os.path.join(train_folder, train_file), 'r', encoding='utf-8') as file:
            train_text = file.read()
            begin = train_text.find('\n', train_text.find('*** START OF THIS PROJECT GUTENBERG EBOOK '))
            end = train_text.find('*** END OF THIS PROJECT GUTENBERG EBOOK')
            train_text = train_text[begin + 1:end].strip()
            text += train_text
    if words:
        text = text.lower()
    sentences = re.split(r'[^\w ]', text)
    text = []
    count = 0
    rhymes = []
    for topic_word in topic_words:
        rhymes.extend(pronouncing.rhymes(topic_word))
    topic_words = topic_words.copy()
    topic_words.extend(rhymes)
    print('!! topic_words=', topic_words)
    for sentence in sentences:
        tokens = re.split(r'\W+', sentence)
        if any([token in topic_words for token in tokens]):
            count += 1
            text.extend(tokens)
    print('!! count=', count)

    word_counts = Counter(text)

    if words:
        missing_words = {word for word in word_counts if word not in words.token_to_index}
        print('!! missing words=', len(missing_words))
        for word in missing_words:
            del word_counts[word]

    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    
    if words:
        words.vectors = torch.index_select(words.vectors, 0, torch.tensor([words.token_to_index[word] for word in sorted_vocab]))
        words.index_to_token = sorted_vocab
        words.token_to_index = vocab_to_int

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text if w in vocab_to_int]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size, pretrained_embeddings):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size, dtype=torch.float),
                torch.zeros(1, batch_size, self.lstm_size, dtype=torch.float))


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def _print_line(
        words: List[str],
        index: int,
        line_number: int
):
    if line_number % flags.stanza_length == 0:
        print()
    print(words[index][0].upper() + (words[index][1:] if len(words[index]) > 1 else ''), end=' ')
    print(' '.join(words[index + 1:index + flags.line_length]))


def _choose_word(
        output,
        top_k,
        words,
        int_to_vocab
) -> int:
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    word = int_to_vocab[choice]
    if word == 'i':
        word = 'I'
    words.append(word)
    return choice


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=10):
    net.eval()
    words = words.copy()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    choice = _choose_word(output, top_k, words, int_to_vocab)

    for index in range(flags.verse_length):
        ix = torch.tensor([[choice]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        choice = _choose_word(output, top_k, words, int_to_vocab)

    line_number = 0
    for index in range(0, len(words), flags.line_length):
        _print_line(words, index, line_number)
        line_number += 1
        if np.random.randint(10) == 1:
            _print_line(words, index, line_number)
            line_number += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    words = FastText(language='en')
    if words:
        flags.lstm_size = flags.embedding_size = words.vectors.shape[1]
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_folder, flags.batch_size, flags.seq_size, words, flags.initial_words)

    net = RNNModule(
        n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size, words.vectors if words else None)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0

    for e in range(flags.n_epochs):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            net.train()

            optimizer.zero_grad()

            x = torch.tensor(x).long().to(device)
            y = torch.tensor(y).long().to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            loss_value = loss.item()

            loss.backward()

            state_h = state_h.detach()
            state_c = state_c.detach()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % flags.train_print == 0:
                print('Epoch: {}/{}'.format(e + 1, flags.n_epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % flags.predict_print == 0:
                predict(
                    device, net, flags.initial_words, n_vocab,
                    vocab_to_int, int_to_vocab)
                torch.save(net.state_dict(),
                           'checkpoint_pt/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()


# In[ ]:




