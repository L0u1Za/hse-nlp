import tokenizers
from tokenizers import normalizers
from tokenizers import pre_tokenizers

from typing import Counter

import torch

class TokenizerBPE(tokenizers.models.Model):
    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.vocab = ['[UNK]', '_sow', '_eow']
        self.token_counter = Counter()
        self.splits = {}
        self.merges = {}

        self.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        self.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=False)]) # pre_tokenizers.ByteLevel()

        super().__init__()
    def _normalize(self, text: str):
        return self.normalizer.normalize_str(text)

    def _pre_tokenize(self, text: str, train=False):
        pre_tokenized = self.pre_tokenizer.pre_tokenize_str(text)
        if (train):
            for word in pre_tokenized:
                self.token_counter[word[0]] += 1
        return pre_tokenized

    def encode(self, sequence):
        normalized = self._normalize(sequence)
        pre_tokenized = self._pre_tokenize(normalized)
        splits = [["_sow", *word, "_eow"] for word, offset in pre_tokenized]
        ids = []

        for j, split in enumerate(splits):
            i = 0
            while (i < len(split)):
                if (i != len(split) - 1 and (split[i], split[i + 1]) in self.merges):
                    split = split[:i] + [self.merges[(split[i], split[i + 1])]] + split[i + 2:]
                else:
                    i += 1
            splits[j] = split
        for split in splits:
            for token in split:
                if token not in self.vocab:
                    ids.append(self.key_to_index['[UNK]'])
                else:
                    ids.append(self.key_to_index[token])

        return {
            "ids": ids
        }

    def decode(self, ids):
        text = ''
        isWord = False
        for cur_id in ids:
            if cur_id == self.key_to_index['_sow']:
                isWord = True
            elif cur_id == self.key_to_index['_eow']:
                text += ' '
                isWord = False
            else:
                text += self.index_to_key[cur_id]
        return text.strip()

    def _collect_vocab_from_splits(self, splits):
        for split in splits.values():
            for char in split:
                if char not in self.vocab:
                    self.vocab.append(char)


    def _merge_splits(self, splits, token):
        final_token = token[0] + token[1]
        for word, split in splits.items():
            if (final_token in word):
                new_split = []
                i = 0
                while (i < len(split)):
                    if (i != len(split) - 1 and (split[i], split[i + 1]) == token):
                        new_split.append(split[i] + split[i + 1])
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split
        return splits

    def _add_freq_pairs(self, splits, pairs_freqs, final_token=None):
        for word, split in splits.items():
            if (final_token):
                if (final_token in word):
                    for i in range(len(split) - 1):
                        pairs_freqs[(split[i], split[i + 1])] += self.token_counter[word] # add frequency of this word from all texts
            else:
                for i in range(len(split) - 1):
                    pairs_freqs[(split[i], split[i + 1])] += self.token_counter[word] # add frequency of this word from all texts
        return pairs_freqs
    def _rem_freq_pairs(self, splits, pairs_freqs, final_token):
        for word, split in splits.items():
            if (final_token in word):
                    for i in range(len(split) - 1):
                        pairs_freqs[(split[i], split[i + 1])] -= self.token_counter[word] # remove frequency of this word from all texts
        return pairs_freqs

    def train(self, corpus):
        _ = [self._pre_tokenize(self._normalize(text), train=True) for text in corpus]

        splits = {}
        for token in self.token_counter.keys():
            splits[token] = [*token]

        self._collect_vocab_from_splits(splits)
        pairs_freqs = self._add_freq_pairs(splits, Counter())

        while (len(self.vocab) < self.max_vocab_size):

            most_freq = pairs_freqs.most_common(1)
            if (most_freq):
                new_token = most_freq[0][0][0] + most_freq[0][0][1]
                self.vocab.append(new_token)
                self.merges[most_freq[0][0]] = new_token

                pairs_freqs = self._rem_freq_pairs(splits, pairs_freqs, new_token) # remove calculated freqs for words there this token is

                splits = self._merge_splits(splits, most_freq[0][0])

                pairs_freqs = self._add_freq_pairs(splits, pairs_freqs, new_token) # add freqs for words there last token was merged
            else: # All corpus in vocab
                break

        self.key_to_index = {}
        for i, token in enumerate(self.vocab):
            self.key_to_index[token] = i
        self.index_to_key = self.vocab

class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad_and_prepare_seq(self, texts, labels):
        batch_ids = [self.tokenizer.encode(text)['ids'] for text in texts]
        max_seq_len = -1
        seq_lengths = []
        for item_ids in batch_ids:
            seq_lengths.append(len(item_ids))
            max_seq_len = max(max_seq_len, len(item_ids))

        inputs = torch.zeros((len(batch_ids), max_seq_len), dtype=torch.long)
        for (i, item_ids) in enumerate(batch_ids):
            for (j, cur_id) in enumerate(item_ids):
                inputs[i][j] = cur_id
        return inputs, torch.tensor(labels, dtype=torch.long), seq_lengths

    def __call__(self, batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs, labels, seq_lengths = self.pad_and_prepare_seq(texts, labels)
        return inputs, labels, seq_lengths

if __name__ == "__main__":
    bpe = TokenizerBPE()
    bpe.train(["Привет пока але", "Привет, нет здравствуй"])
    print(bpe.decode(bpe.encode('але а что такое здравствуй')['ids']))