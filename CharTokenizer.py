""" Simple tokenizer that splits texts into characters. """
from io import open

class CharTokenizer():
    def __init__(self, path_vocab):
        """Make tokenizer. 

        Args:

        """
        self.path_vocab = path_vocab
        self.vocab = self._init_vocab(path_vocab)
        self.char2count = {}
        with open(path_vocab, encoding="utf-8") as f:
            for line in f:
                elems = line.rstrip().split("\t")
                assert len(elems) == 2
                char = elems[0]
                count = int(elems[1])
                self.char2count[char] = count                
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        return

    
    def _init_vocab(self, path_vocab):
        vocab = {}
        vocab["[PAD]"] = len(vocab)
        vocab["[UNK]"] = len(vocab)
        vocab["[MASK]"] = len(vocab)
        vocab["[CLS]"] = len(vocab)
        vocab["[SEP]"] = len(vocab)
        vocab[" "] = len(vocab)
        return vocab

    
    def update_vocab(self, text):
        """ Update vocab and character frequencies. """
        for char in text:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
            if char not in self.char2count:
                self.char2count[char] = 0
            self.char2count[char] += 1
        return

    
    def trim_vocab(self, min_freq):
        if min_freq < 1:
            return
        if not self.char2count:
            msg = "Provide training data when initializing tokenizer if you want to then trim the vocab."
            raise NotImplementedError(msg)

        new_vocab = self._init_vocab(self.path_vocab)
        new_char2count = {}
        for char in self.char2count.keys():
            if char not in new_vocab and self.char2count[char] >= min_freq:
                new_vocab[char] = len(new_vocab)
                new_char2count[char] = self.char2count[char]
        self.vocab = new_vocab
        self.char2count = new_char2count
        return

    
    def tokenize(self, chars):
        return [c if c in self.vocab else "[UNK]" for c in chars]

    
    def convert_tokens_to_ids(self, chars):
        return [self.vocab[c] if c in self.vocab else self.vocab["[UNK]"] for c in chars]
    


