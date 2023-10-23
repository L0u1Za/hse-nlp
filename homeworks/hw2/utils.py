import tokenizers
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.processors import TemplateProcessing

from typing import Counter
from typing import DefaultDict

class TokenizerBPE(tokenizers.models.Model):
    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.vocab = ['[UNK]']
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
        ids = []
        for word, offset in pre_tokenized:
            if not (word in self.key_to_index):
                ids.append(self.key_to_index['[UNK]'])
            else:
                ids.append(self.key_to_index[word])
        return {
            "ids": ids
        }

    def decode(self, ids):
        text = ''
        for id in ids:
            text += ' ' + self.index_to_key[id]
        return text.strip()

    def _collect_vocab_from_splits(self, splits):
        for split in splits.values():
            for char in split:
                if char not in self.vocab:
                    self.vocab.append(char)


    def _merge_splits(self, splits, token):
        for word, split in splits.items():
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

    def _calc_freq_pairs(self, splits):
        pairs_freqs = Counter()
        for word, split in splits.items():
            for i in range(len(split) - 1):
                pairs_freqs[(split[i], split[i + 1])] += self.token_counter[word] # add frequency of this word from all texts
        return pairs_freqs

    def train(self, corpus):
        _ = [self._pre_tokenize(self._normalize(text), train=True) for text in corpus]
        splits = {}
        for token in self.token_counter.keys():
            splits[token] = [*token]

        self._collect_vocab_from_splits(splits)

        while (len(self.vocab) < self.max_vocab_size):
            pairs_freqs = self._calc_freq_pairs(splits)

            most_freq = pairs_freqs.most_common(1)
            if (most_freq):
                new_token = most_freq[0][0][0] + most_freq[0][0][1]
                self.vocab.append(new_token)
                self.merges[most_freq[0][0]] = new_token
                splits = self._merge_splits(splits, most_freq[0][0])
            else: # All corpus in vocab
                break
        self.splits = splits

        self.key_to_index = {}
        for i, token in enumerate(self.vocab):
            self.key_to_index[token] = i
        self.index_to_key = self.vocab

t = TokenizerBPE()
t.train(["Привет але але", "Привет пока, а че"])

print(t.decode(t.encode("Привет пока абабаопа")['ids']))