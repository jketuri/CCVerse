#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from argparse import Namespace
from collections import Counter
import os
import re
from typing import Dict, List, Set

import numpy as np

import language_check
import pronouncing
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.word_to_vector import BPEmb, FastText

flags = Namespace(
    train_folder='data',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['i', 'love'],
    predict_top_k=5,
    checkpoint_path='checkpoint',
    n_epochs=150,
    verse_length=100,
    train_print=100,
    predict_print=100,
    line_length=8,
    stanza_length=4
)


def _update_words(
        text_tokens: List[str],
        words: FastText
):
    word_counts = Counter(text_tokens)

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
    return int_to_vocab, vocab_to_int, n_vocab


def get_data_from_files(
        train_folder: str,
        batch_size: int,
        seq_size: int,
        words: FastText,
        topic_words: List[str]
):
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
    text_tokens = []
    for sentence in sentences:
        tokens = re.split(r'\W+', sentence)
        text_tokens.extend(tokens)
        text_tokens.append('.')

    int_to_vocab, vocab_to_int, n_vocab = _update_words(text_tokens, words)
    print('Vocabulary size', n_vocab)

    print('!! topic_words=', topic_words)
    rhymes = []
    for topic_word in topic_words:
        rhymes_for_topic_word = pronouncing.rhymes(topic_word)
        print('!! rhymes_for_topic_word=', rhymes_for_topic_word)
        rhyme_set = set()
        for rhyme in rhymes_for_topic_word:
            if rhyme in words.token_to_index and rhyme not in rhyme_set:
                rhymes.append(rhyme)
                rhyme_set.add(rhyme)
    print('!! rhymes=', rhymes)
    print('!! number of rhymes=', len(rhymes))

    clusters = KMeans(n_clusters=int(n_vocab / 100)).fit_predict(X=words.vectors)
    print("number of estimated clusters : %d" % len(np.unique(clusters)))
    theme_words = set()
    cluster = clusters[words.token_to_index[topic_words[-1]]]
    for cluster_idx in range(len(clusters)):
        if clusters[cluster_idx] == cluster:
            theme_words.add(words.index_to_token[cluster_idx])
    print('!! theme_words=', theme_words)
    print('!! number of theme_words=', len(theme_words))
    select_words = theme_words.copy()
    select_words.update(topic_words)
    select_words.update(rhymes)
    print('!! select_words=', select_words)

    text_tokens = []
    sentences = re.split(r'[^\w ]', text)
    for sentence in sentences:
        tokens = re.split(r'\W+', sentence)
        if any([token in select_words for token in tokens]):
            text_tokens.extend(tokens)
            text_tokens.append('.')

    int_to_vocab, vocab_to_int, n_vocab = _update_words(text_tokens, words)
    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text_tokens if w in vocab_to_int]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, theme_words


def get_batches(
        in_text,
        out_text,
        batch_size,
        seq_size
):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class RNNModule(
        nn.Module
):

    def __init__(
            self,
            n_vocab,
            seq_size,
            embedding_size,
            lstm_size,
            pretrained_embeddings
    ):
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

    def forward(
            self,
            x,
            prev_state
    ):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(
            self,
            batch_size
    ):
        return (torch.zeros(1, batch_size, self.lstm_size, dtype=torch.float),
                torch.zeros(1, batch_size, self.lstm_size, dtype=torch.float))


def get_loss_and_train_op(
        net,
        lr=0.001
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


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


def predict(
        tool,
        device,
        net,
        words,
        n_vocab,
        vocab_to_int,
        int_to_vocab,
        theme_words: Set,
        top_k=5
):
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

    verse = ' '.join(words)
    sentences = re.split(r'[^\w ]', verse)
    for sentence in sentences:
        tokens = re.split(r'\W+', sentence)
        if any([theme_word in tokens for theme_word in theme_words]):
            matches = tool.check(sentence)
            sentence1 = language_check.correct(sentence, matches)
            print(sentence1)
            if np.random.randint(10) == 1:
                print(sentence1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    words = FastText(language='en')
    if words:
        flags.lstm_size = flags.embedding_size = words.vectors.shape[1]
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, theme_words = get_data_from_files(
        flags.train_folder, flags.batch_size, flags.seq_size, words, flags.initial_words)

    net = RNNModule(
        n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size, words.vectors if words else None)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    tool = language_check.LanguageTool('en-US')

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
                    tool, device, net, flags.initial_words, n_vocab,
                    vocab_to_int, int_to_vocab, theme_words)
                torch.save(net.state_dict(),
                           'checkpoint_pt/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()


# In[ ]:




