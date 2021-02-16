""" Dataset classes for training and evaluating language detectors and language identifiers. """

import os, random, logging, glob
from copy import deepcopy
from itertools import chain
from iteround import saferound
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import compute_split_sampling_probs, compute_sample_sizes, load_or_make_lang2freq
from utils import sort_examples_by_text_length, line_to_data
from comp_utils import ALL_LANGS
from scorer import compute_fscores

# Label used in BertForLM to indicate a token is not masked.
NO_MASK_LABEL = -100

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def yield_line(from_file):
    # Read line from file, wrapping around if necessary
    line = from_file.readline()
    if line:
        return line
    else:
        from_file.seek(0)
        line = from_file.readline()
        if not line:
            msg = "file is empty"
            raise RuntimeError(msg)
        return line


def make_tensor(input_array):
    if input_array is None:
        return torch.empty(0)
    else:
        return torch.tensor(input_array)


def mask_random_tokens(tokens, tokenizer):
    """Mask some random tokens for masked language modeling with
    probabilities as in the original BERT paper.

    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction

    """
    output_label = []
    # Copy tokens so that we don't modify the input list.
    tokens = deepcopy(tokens)
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            output_label.append(NO_MASK_LABEL)

    return tokens, output_label


class InputExampleForPretraining(object):
    """A single training/test example for masked language modeling and sentence pair classification. """

    def __init__(self, guid, tokens, spc_tokens=None, spc_label=None):
        """Constructs a InputExample for pre-training (MLM and optional SPC).

        Params:
        - guid: Unique id for the example.
        - tokens: list of strings. The tokens of the text sample (i.e. a query in the case of SPC).
        - spc_tokens: (Optional) list of strings. The tokens of a candidate for SPC (i.e. another text sample).
        - spc_label: (Optional) 1 or 0. Label (1 if spc_tokens and tokens are a positive example, 0 otherwise).  

        """
        self.guid = guid
        self.tokens = tokens
        self.spc_tokens = spc_tokens
        self.spc_label = spc_label
        

class InputFeaturesForPretraining(object):
    """A single set of features for pretraining (i.e. masked language
    modeling and optionally sentence pair classification).

    Note: the features for SPC can be None

    Args:
    - input_ids: input IDs
    - input_ids_masked: input IDs in which some tokens have been replaced with [MASK] for MLM
    - input_mask: input attention mask (1 for tokens, 0 for padding)
    - segment_ids: token type (segment) IDs
    - seq_len: real sequence length of input tokens (excluding padding, but including special token CLS).
    - lm_label_ids: language model label IDs
    - spc_input_ids: input IDs of a candidate for SPC
    - spc_input_mask: input mask of a candidate for SPC
    - spc_segment_ids: token type (segment) IDs of a candidate for SPC
    - spc_seq_len: real sequence length of input tokens of candidate for SPC
    - spc_label: integer label of (query, candidate) pair for SPC.

    """
    def __init__(self, input_ids, input_ids_masked, input_mask, segment_ids, seq_len, lm_label_ids, spc_input_ids, spc_input_mask, spc_segment_ids, spc_seq_len, spc_label):
        self.input_ids = input_ids
        self.input_ids_masked = input_ids_masked
        self.input_mask = input_mask 
        self.segment_ids = segment_ids 
        self.seq_len = seq_len 
        self.lm_label_ids = lm_label_ids 
        self.spc_input_ids = spc_input_ids 
        self.spc_input_mask = spc_input_mask 
        self.spc_segment_ids = spc_segment_ids 
        self.spc_seq_len = spc_seq_len 
        self.spc_label = spc_label 


class DatasetForPretraining(Dataset):

    """Dataset for pre-training, with MLM and optionally SPC. 

    Creates an index of the training data, as storing all the text
    examples would take too much memory. Data is then sampled using
    this index.

    """

    def __init__(self, dir_data, tokenizer, max_seq_len, max_sample_size=100000, include_spc=False, sampling_alpha=1.0, weight_relevant=1.0, encoding="utf-8", seed=None, verbose=False):
        super().__init__()
        assert sampling_alpha >= 0 and sampling_alpha <= 1
        self.dir_data = dir_data                 # Directory containing training files with names matching <lang>.train.
        self.tokenizer = tokenizer               # Tokenizer
        self.max_seq_len = max_seq_len           # Maximum sequence length. Includes CLS token
        self.max_sample_size = max_sample_size   # Max examples sampled each time refresh() is called.
        self.include_spc = include_spc           # Include features for sentence pair classification
        self.sampling_alpha = sampling_alpha     # Damping factor alpha
        self.weight_relevant = weight_relevant   # Sampling weight of relevant examples wrt irrelevant examples
        self.encoding = encoding
        self.verbose = verbose
        self.sample_counter = 0  # Counts calls to __getitem__
        self.label_list = []
        self.lang2id = {}    # Maps to indices in label_list
        self.lang2path = {}  # Maps to path of training file
        self.sample_sizes = [] # Per-language sample sizes
        self.data = [] # List of texts (if MLM only) or (text 1, text 2, SPC label) tuples.

        # Map languages to the training file paths. Make sure dir_data
        # contains either one training file for each language in
        # ALL_LANGS, or a single file called "unk.train"
        train_file_paths = glob.glob(os.path.join(dir_data, "*.train"))
        if len(train_file_paths) > 1:
            for path in train_file_paths:
                lang = path[-9:-6]
                self.lang2path[lang] = path
            assert set(self.lang2path.keys()) == set(ALL_LANGS)
            self.label_list = sorted(ALL_LANGS)
            self.lang2id = {x:i for i,x in enumerate(self.label_list)}
        else:
            assert train_file_paths[0][-9:] == "unk.train"
            assert self.include_spc == False

            # If all we have is unk.train, then we just load all the
            # data and sort by length
            with open(train_file_paths[0], 'r', encoding=self.encoding) as f:
                for line in f:
                    text, _ = line_to_data(line, False, strict=True)
                    self.data.append(text)
            self.data = sort_examples_by_text_length(self.data, text_ix=0)
            return
        
        # Seed RNG
        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Load or make map lang2freq (and optionally lang2lens)
        if include_spc:
            self.lang2freq, lang2lens = load_or_make_lang2freq(dir_data, include_lengths=True)
            logger.info("Making data structures used to find lines of similar length")
            self.lang2lenargsort = {k:np.argsort(v).tolist() for k,v in lang2lens.items()}
            self.lang2lenargsortinv = {k:np.argsort(v).tolist() for k,v in self.lang2lenargsort.items()}
            self.look_forward = True
        else:
            self.lang2freq = load_or_make_lang2freq(dir_data, include_lengths=False)
            
        # Compute sample sizes
        sample_probs = compute_split_sampling_probs(self.lang2freq,
                                                    sampling_alpha=self.sampling_alpha,
                                                    weight_relevant=self.weight_relevant)
        max_sample_sizes = np.array([self.lang2freq[k] for k in self.label_list],
                                    dtype=np.int)
        self.sample_sizes = compute_sample_sizes(sample_probs, max_sample_sizes, self.max_sample_size, scale_total=True)
        self.total_sample_size = sum(self.sample_sizes)
        logger.info("Stats on sample sizes:")
        logger.info("- Min: %d" % min(self.sample_sizes))
        logger.info("- Max: %d" % max(self.sample_sizes))
        logger.info("- Sum: %d" % sum(self.sample_sizes))

        # Store a random order of line numbers for each language
        logger.info("Preparing maps for random sampling")
        self.lang2randorder = {}
        for lang in self.lang2path:
            rand_order = list(range(self.lang2freq[lang]))
            np.random.shuffle(rand_order)
            self.lang2randorder[lang] = rand_order[:]
        self.lang2linepointer = {k:0 for k in self.lang2freq.keys()}

        # Sample data
        self.resample()
        return


    def __len__(self):
        return self.total_sample_size
    
    
    def resample(self):
        """ Sample pre-training data, populate self.data, and sort by length. """
        logger.info("Sampling examples")
        if self.include_spc:
            self._resample_with_spc()
        else:
            if len(self.lang2path) == 1:
                msg = "Can not sample if dataset only contains unk.train"
                raise RuntimeError(msg)
            self._resample_without_spc()
        if self.include_spc:
            logger.info("Length of shortest text: %d" % len(self.data[0][0]))
            logger.info("Length of longest text: %d" % len(self.data[-1][0]))                        
        else:
            logger.info("Length of shortest text: %d" % len(self.data[0]))
            logger.info("Length of longest text: %d" % len(self.data[-1]))            
        return


    def _resample_without_spc(self):
        self.data = [] # List of texts
        for lang, ss in zip(self.label_list, self.sample_sizes):
            logger.info("  Sampling {} (n={})".format(lang, ss))
            if ss == 0:
                continue
            lines_to_retrieve = [self._get_random_line_ix(lang) for i in range(ss)]
            lines_to_retrieve = sorted(lines_to_retrieve)
            path = self.lang2path[lang]
            with open(path, 'r', encoding=self.encoding) as f:
                for line_ix, line in enumerate(f):
                    if line_ix == lines_to_retrieve[0]:
                        text, _ = line_to_data(line, False, strict=True)
                        assert text is not None
                        self.data.append(text)
                        _ = lines_to_retrieve.pop(0)
                        if len(lines_to_retrieve) == 0:
                            break
        # Sort by length
        self.data = sort_examples_by_text_length(self.data, text_ix=0)
        return
    
    
    def _resample_with_spc(self):
        self.data = [] # List of (query text, candidate text, SPC label) tuples

        # First, sample from languages whose sample size is 1
        logger.info("Sampling from languages whose sample size is 1")
        for lang, ss in zip(self.label_list, self.sample_sizes):
            if ss == 1:
                # Retrieve one line
                line_ix = self._get_random_line_ix(lang)
                path = self.lang2path[lang]
                with open(path, 'r', encoding=self.encoding) as f:
                    for i in range(line_ix):
                        # Skip line
                        next(f)
                    line = f.readline()
                    text, _ = line_to_data(line, False, strict=True)
                    assert text is not None
                    # This has to be a negative examples for SPC, as
                    # we only sample one text for this language
                    self.data.append([text,None,0,lang])
                    
        # Check how many negative examples we already have
        nb_neg = len(self.data)
        logger.info("Nb negatives obtained from languages whose sample size is 1: %d" % nb_neg)
        
        # Compute negative sampling probability such that we will
        # sample approximately the same number of positive and
        # negative examples for SPC
        remaining_ss = self.total_sample_size - nb_neg
        nb_neg_left = (self.total_sample_size // 2) - nb_neg
        neg_sampling_prob = nb_neg_left / remaining_ss
        logger.info("Neg sampling prob (s.t. we will get a balanced mix of pos and neg): %f" % neg_sampling_prob)

        # Sample texts from languages whose sample size is greater than 1
        logger.info("Sampling from languages whose sample is greater than 1")        
        for lang, ss in zip(self.label_list, self.sample_sizes):
            if ss < 2:
                continue
            logger.info("  Sampling {} (n={})".format(lang, ss))
            lines_to_retrieve = []
            spc_labels = []
            for i in range(ss):
                line_ix = self._get_random_line_ix(lang)
                lines_to_retrieve.append(line_ix)
                # Flip a (potentially biased) coin to determine
                # whether we will use this text in a positive or
                # negative example for SPC
                spc_label = 1 if random.random() > neg_sampling_prob else 0
                spc_labels.append(spc_label)
            # Find lines of similar length to create positive examples for SPC
            other_lines_to_retrieve = [None] * ss
            for (i, (line_ix, label)) in enumerate(zip(lines_to_retrieve, spc_labels)):
                if label == 1:
                    other_lines_to_retrieve[i] = self._find_line_with_similar_length(lang, line_ix)
            # Initialize list of triples we will add to self.data
            data_sub = [[None,None,y] for y in spc_labels]

            # Create map from line indices to a list of coordinates in
            # data_sub. Note that a line index can map to 1 or 2
            # coordinate tuples, or in a rare case, 3 (i.e. when the
            # sample size for a given language equals the number of
            # available training examples, the second-longest sentence
            # will be sampled twice as the candidate text).
            lix_to_coord = {}
            for (row_ix, (lix, label, lix2)) in enumerate(zip(lines_to_retrieve, spc_labels, other_lines_to_retrieve)):
                if lix not in lix_to_coord:
                    lix_to_coord[lix] = []
                lix_to_coord[lix].append((row_ix, 0))
                if label == 1:
                    if lix2 not in lix_to_coord:
                        lix_to_coord[lix2] = []
                    lix_to_coord[lix2].append((row_ix,1))

            # Retrieve lines and populate first 2 columns in data_sub
            lines_to_retrieve = sorted(set(lines_to_retrieve + [x for x in other_lines_to_retrieve if x is not None]))
            path = self.lang2path[lang]
            with open(path, 'r', encoding=self.encoding) as f:
                for line_ix, line in enumerate(f):
                    if line_ix == lines_to_retrieve[0]:
                        text, _ = line_to_data(line, False, strict=True)
                        assert text is not None
                        coords = lix_to_coord[line_ix]
                        assert len(coords) > 0
                        assert len(coords) < 4
                        for (row,col) in coords:
                            data_sub[row][col] = text
                        _ = lines_to_retrieve.pop(0)
                        if len(lines_to_retrieve) == 0:
                            break

            # Add triples to self.data (add lang temporarily)
            for (text1, text2, label) in data_sub:
                assert text1 is not None
                assert label in [0,1]
                if label == 1:
                    assert text2 is not None
                self.data.append([text1, text2, label, lang])
            
        # Check data
        assert len(self.data) == self.total_sample_size
        pos_count = sum(x[2] for x in self.data)
        pct = 100 * pos_count / self.total_sample_size
        logger.info("Nb positive examples: %d/%d (%.1f%%)" % (pos_count, self.total_sample_size, pct))

        # Sort by length
        logger.info("Sorting by length")
        self.data = sort_examples_by_text_length(self.data, text_ix=0)

        # Now get other text for negative examples
        logger.info("Getting texts to populate negative examples")
        for i in range(self.total_sample_size):
            label = self.data[i][2]
            t1_lang = self.data[i][3]
            if label == 0:
                down_indices = range(i+1, self.total_sample_size)
                up_indices = range(i-1, -1, -1)
                done = False
                for j in chain(down_indices, up_indices):
                    t1_lang_other = self.data[j][3]
                    if t1_lang_other != t1_lang:
                        t1_other = self.data[j][0]                        
                        self.data[i][1] = t1_other
                        done = True
                        break
                if not done:
                    msg = "No text in other language found"
                    raise RuntimeError(msg)                            

        # Remove column containing language, which is no longer necessary
        self.data = [x[:3] for x in self.data]
        
        # Log some stats
        len_diffs = [len(x[1])-len(x[0]) for x in self.data]        
        logger.info("Mean length diff: %f" % np.mean(len_diffs))
        logger.info("Mean absolute length diff: %f" % np.mean(np.abs(len_diffs)))
        return

    
    def _find_line_with_similar_length(self, lang, line_ix):
        ix = self.lang2lenargsortinv[lang][line_ix]
        if ix == 0:
            return self.lang2lenargsort[lang][1]
        elif ix == (self.lang2freq[lang] - 1):
            return self.lang2lenargsort[lang][-2]
        else:
            if self.look_forward:
                return self.lang2lenargsort[lang][ix+1]
            else:
                return self.lang2lenargsort[lang][ix-1]
            # Flip switch so we look in opposite direction next time
            self.look_forward = not self.look_forward

        
    def _get_random_line_ix(self, lang):
        line_ix = self.lang2randorder[lang][self.lang2linepointer[lang]]
        self.lang2linepointer[lang] += 1
        if self.lang2linepointer[lang] == self.lang2freq[lang]:
            self.lang2linepointer[lang] = 0
        return line_ix
    
            
    def __getitem__(self, item):
        """ Get an example for pre-training.

        Return: list of tensors, one for each feature in InputFeaturesForPretraining.        

        """
        # Increment sample counter
        guid = self.sample_counter
        self.sample_counter += 1

        # Get an example
        if self.include_spc:
            [text1, text2, spc_label] = self.data[item]
            tokens = self.tokenizer.tokenize(text1)
            spc_tokens = self.tokenizer.tokenize(text2)
            example = InputExampleForPretraining(guid=guid,
                                                 tokens=tokens,
                                                 spc_tokens=spc_tokens,
                                                 spc_label=spc_label)
            
        else:
            text = self.data[item]
            tokens = self.tokenizer.tokenize(text)
            example = InputExampleForPretraining(guid=guid,
                                                 tokens=tokens,
                                                 spc_tokens=None,
                                                 spc_label=None)
        features = self._convert_example_to_features(example)
        tensors = [make_tensor(features.input_ids),
                   make_tensor(features.input_ids_masked),
                   make_tensor(features.input_mask),
                   make_tensor(features.segment_ids),
                   make_tensor(features.seq_len),
                   make_tensor(features.lm_label_ids),
                   make_tensor(features.spc_input_ids),
                   make_tensor(features.spc_input_mask),
                   make_tensor(features.spc_segment_ids),
                   make_tensor(features.spc_seq_len),                   
                   make_tensor(features.spc_label)]
        return tensors


    def _convert_example_to_features(self, example):
        """Convert an InputExampleForPretraining to InputFeaturesForPretraining.

        :param example: InputExampleForPretraining.

        :return: InputFeaturesForPretraining

        """
        guid = example.guid
        tokens = example.tokens
        spc_tokens = example.spc_tokens
        spc_label = example.spc_label

        # Truncate sequences if necessary. Account for [CLS] by subtracting 1.
        tokens = tokens[:self.max_seq_len-1]
        if spc_tokens is not None:
            spc_tokens = spc_tokens[:self.max_seq_len-1]
        
        # Mask tokens for MLM
        tokens_masked, lm_label_ids = mask_random_tokens(tokens, self.tokenizer)
        
        # Add [CLS], store sequence length
        tokens = ["[CLS]"] + tokens
        tokens_masked = ["[CLS]"] + tokens_masked        
        lm_label_ids = [NO_MASK_LABEL] + lm_label_ids
        seq_len = len(tokens)
        if spc_tokens is not None:
            spc_tokens = ["[CLS]"] + spc_tokens
            spc_seq_len = len(spc_tokens)
        else:
            spc_tokens = None
            spc_seq_len = None
            
        # Get input token IDs, add zero-padding
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * (self.max_seq_len - seq_len)
        input_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)
        input_ids_masked += [0] * (self.max_seq_len - seq_len)
        lm_label_ids += [NO_MASK_LABEL] * (self.max_seq_len - seq_len)        
        if spc_tokens is not None:
            spc_input_ids = self.tokenizer.convert_tokens_to_ids(spc_tokens)
            spc_input_ids += [0] * (self.max_seq_len - spc_seq_len)
        else:
            spc_input_ids = None
            
        # Make input mask (1 for real tokens and 0 for padding tokens) and segment IDs
        input_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)
        segment_ids = [0] * self.max_seq_len
        if spc_tokens is not None:
            spc_input_mask = [1] * spc_seq_len + [0] * (self.max_seq_len - spc_seq_len)
            spc_segment_ids = [0] * self.max_seq_len
        else:
            spc_input_mask = None
            spc_segment_ids = None
            
        # Check data
        assert len(input_ids) == self.max_seq_len
        assert len(input_ids_masked) == self.max_seq_len
        assert len(lm_label_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        if spc_tokens is not None:
            assert len(spc_input_ids) == self.max_seq_len        
            assert len(spc_input_mask) == self.max_seq_len
            assert len(spc_segment_ids) == self.max_seq_len
        
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("  guid: {}".format(example.guid))
            logger.info("  tokens: {}".format(tokens))
            logger.info("  tokens_masked: {}".format(tokens_masked))            
            logger.info("  input_ids: {}".format(input_ids))
            logger.info("  input_ids_masked: {}".format(input_ids_masked))
            logger.info("  lm_label_ids: {}".format(lm_label_ids))
            logger.info("  input_mask: {}".format(input_mask))
            logger.info("  segment_ids: {}".format(segment_ids))
            logger.info("  seq_len: {}".format(seq_len))
            logger.info("  spc_tokens: {}".format(spc_tokens))
            logger.info("  spc_input_ids: {}".format(spc_input_ids))
            logger.info("  spc_input_mask: {}".format(spc_input_mask))
            logger.info("  spc_segment_ids: {}".format(spc_segment_ids))
            logger.info("  spc_seq_len: {}".format(spc_seq_len))
            logger.info("  spc_label: {}".format(spc_label))
            
        # Get features
        features = InputFeaturesForPretraining(input_ids,
                                               input_ids_masked,
                                               input_mask,
                                               segment_ids,
                                               seq_len,
                                               lm_label_ids,
                                               spc_input_ids,
                                               spc_input_mask,
                                               spc_segment_ids,
                                               spc_seq_len,
                                               spc_label)
        return features

        
class InputExampleForClassification(object):
    """A single training/test example for language identification
    (i.e. multi-class classification).

    """

    def __init__(self, guid, tokens, label_id=None, candidate_classes=None, pos_cand_id=None):
        """Constructs a InputExample.

        Params:
        - guid: Unique id for the example.
        - tokens: list of strings. The tokens.
        - label_id: (optional) label id
        - candidate_classes: (optional) list of integers. The classes
          to evaluate for this example. This can be None, a list
          containing only the correct class (if no subsampling of
          negative classes was applied for training purposes, or a
          list containing any number of class IDs.
        - pos_cand_id: (optional) int indicating index of candidate
          class that is the true class

        """
        self.guid = guid
        self.tokens = tokens
        self.label_id = label_id
        self.candidate_classes = candidate_classes
        self.pos_cand_id = pos_cand_id
        

class InputFeaturesForClassification(object):
    """A single set of features of data for language identification
    (i.e. multi-class classification).

    Params:
    - input_ids: list input token IDs
    - input_mask: list containing input mask 
    - segment_ids: list containing token type (segment) IDs
    - seq_len: actual sequence length of input tokens (excluding
      padding, but including special token CLS).
    - label_id: (optional) label id
    - candidate_classes: (optional) list containing subsampled class
      IDs to evaluate (containing only the true class if no
      subsampling of negative classes was applied for training
      purposes)
    - pos_cand_id: (optional) int indicating index of candidate class
      that is the true class

    """

    def __init__(self, input_ids, input_mask, segment_ids, seq_len, label_id=None, candidate_classes=None, pos_cand_id=None):
        self.input_ids = input_ids 
        self.input_mask = input_mask 
        self.segment_ids = segment_ids 
        self.seq_len = seq_len
        self.label_id = label_id 
        self.candidate_classes = candidate_classes 
        self.pos_cand_id = pos_cand_id

        
class DatasetForClassification(Dataset):
    """A class for training or evaluating language identification
    (i.e. multi-class classification).

    """

    def __init__(self, data):
        """Init

        Params: 
        - data: list where each item contains a text, an optional true
        class ID, a list of candidate class IDs (containing only the
        true class ID if no subsampling of negative classes is applied),
        and a list of binary labels for each of the candidates.

        """
        # Init parent class
        super().__init__()
        self.data = data 
        self.sample_counter = 0 # Counts calls to __getitem__

        # Map languages to integer IDs
        self.label_list = sorted(ALL_LANGS)
        self.lang2id = {x:i for i,x in enumerate(self.label_list)}


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, item):
        text, label_id, cands, pos_cand_id = self.data[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens = self.tokenizer.tokenize(text)
        example = InputExampleForClassification(example_id, tokens, label_id, candidate_classes=cands, pos_cand_id=pos_cand_id)
        features = self._convert_example_to_features(example)
        tensors = [torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids),
                   torch.tensor(features.seq_len)]
        if features.label_id is None:
            tensors.append(torch.empty(0))
        else:
            tensors.append(torch.tensor(features.label_id).long())
        if features.candidate_classes is None:
            tensors.append(torch.empty(0))
        else:
            tensors.append(torch.tensor(features.candidate_classes).long())
        if features.pos_cand_id is None:
            tensors.append(torch.empty(0))
        else:
            tensors.append(torch.tensor(features.pos_cand_id).long())
        return tensors


    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for classification.
        
        :param example: InputExampleForClassification.

        :return: InputFeaturesForClassification.

        """
        tokens = example.tokens
        label_id = example.label_id
        candidate_classes = example.candidate_classes
        pos_cand_id = example.pos_cand_id
        
        # Truncate sequence if necessary. Account for [CLS] by subtracting 1.
        tokens = tokens[:self.max_seq_len-1]

        # Add CLS
        tokens = ["[CLS]"] + tokens
        seq_len = len(tokens)
        
        # Get input token IDs (unpadded)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
        # Zero-pad input token IDs
        input_ids += [0] * (self.max_seq_len - len(tokens))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask = [1] * len(tokens) + [0] * (self.max_seq_len - len(tokens))
    
        # Make segment IDs (padded)
        segment_ids = [0] * self.max_seq_len

        # Check data
        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("  guid: {}".format(example.guid))
            logger.info("  tokens: {}".format(tokens))
            logger.info("  input_ids: {}".format(input_ids))
            logger.info("  input_mask: {}".format(input_mask))
            logger.info("  segment_ids: {}".format(segment_ids))
            logger.info("  seq_len: {}".format(seq_len))            
            logger.info("  label_id: {}".format(label_id))
            logger.info("  label: {}".format(self.label_list[label_id]))
            logger.info("  candidate class IDs: {}".format(candidate_classes))
            logger.info("  positive candidate ID: {}".format(pos_cand_id))

        # Get features
        features = InputFeaturesForClassification(input_ids=input_ids,
                                                  input_mask=input_mask,
                                                  segment_ids=segment_ids,
                                                  seq_len=seq_len,
                                                  label_id=label_id,
                                                  candidate_classes=candidate_classes,
                                                  pos_cand_id=pos_cand_id)
        return features

    
class DatasetForTrainingClassifier(DatasetForClassification):
    """ A class for training language identification (i.e. multi-class classification). """
    
    def __init__(self, dir_data, tokenizer, sampling_probs, sample_size=100000, max_seq_len=256, encoding="utf-8", seed=None, verbose=False):
        """ Init

        Params:
        - dir_data: path of directory containing training files (one per language)
        - tokenizer:
        - sampling_probs: numpy array of language sampling probabilities
        - sample_size: number of examples sampled from training files (because we can't store them all in memory)
        - max_seq_len: maximum sequence length (including CLS)

        """
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.sampling_probs = sampling_probs
        self.sample_size = sample_size
        self.max_seq_len = max_seq_len # Includes CLS token
        self.encoding = encoding
        self.verbose = verbose
        self.lang2freq = load_or_make_lang2freq(self.dir_data)
        self.label_list = sorted(ALL_LANGS)
        self.lang2id = {x:i for i,x in enumerate(self.label_list)}
        self.max_sample_sizes = np.array([self.lang2freq[k] for k in self.label_list], dtype=np.int)
        
        # Seed RNG
        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Open all training files
        self.lang2file = {}
        for lang in self.label_list:
            path = os.path.join(self.dir_data, "%s.train" % lang)            
            self.lang2file[lang] = open(path, encoding=self.encoding)
            
        # Sample data
        data = self._sample_data(sampling_probs, verbose=True)

        # Init parent
        super().__init__(data)

        
    def resample(self, sampling_probs):
        """ Overwrite self.data with new samples. """
        self.data = self._sample_data(sampling_probs, verbose=False)


    def _sample_data(self, sampling_probs, verbose=False):
        """Sample examples. Compute class-wise sample sizes given sampling
        probs, then get samples sequentially from training files
        (assuming they are shuffled).

        Params:
        - Sampling_probs: if None, we assume uniform

        """
        # Compute sample sizes
        if sampling_probs is None: # Assume uniform
            sampling_probs = np.ones(len(self.label_list), dtype=np.float)
        if sampling_probs.sum() != 1:
            sampling_probs /= sampling_probs.sum() # Normalize
        expected_sample_sizes = sampling_probs * self.sample_size
        sample_sizes = saferound(expected_sample_sizes, 0, 'difference')
        sample_sizes = [int(x) for x in sample_sizes]
        assert sum(sample_sizes) == self.sample_size
        
        if verbose:
            logger.info("Stats on sample sizes:")
            logger.info("- Min: %d" % min(sample_sizes))
            logger.info("- Max: %d" % max(sample_sizes))
            logger.info("- Sum: %d" % sum(sample_sizes))

        # Sample examples
        logger.info("Sampling examples")
        data = []
        for lang, ss in zip(self.label_list, sample_sizes):
            langid = self.lang2id[lang]
            logger.info("  Sampling %d examples of '%s'" % (ss, lang))
            if ss == 0:
                continue
            for _ in range(ss):
                line = yield_line(self.lang2file[lang])
                text = line.rstrip()
                data.append((text, langid, [langid], 0))
        assert len(data) == self.sample_size
        np.random.shuffle(data)
        return data
        
        
    def subsample_neg_classes(self, nb_samples, sampling_probs=None):
        """Subsample the negative classes to be evaluated for each
        observation. If sampling probabilities are provided, these are
        used, otherwise a uniform distribution is assumed.

        Params:
        - nb_samples: int
        - sampling_probs: (optional) a 2-D array of sampling
          probabilities, shape (nb_classes, nb_classes), with zero
          diagonal, where every row is the sampling probability
          distribution for a given target class.

        """
        assert nb_samples < len(self.label_list)
        if sampling_probs is not None:
            assert type(sampling_probs) == np.ndarray
            assert sampling_probs.shape == (len(self.label_list), len(self.label_list))
            assert sampling_probs.diagonal().sum() == 0
            if not np.all(np.abs(np.sum(sampling_probs, 1)-1) < 1e-5):
                msg = "rows in sampling_probs should sum to 1"
                raise ValueError(msg)
        
        # Get true classes
        true_classes = [x[1] for x in self.data]
        
        # Subsample negative classes
        neg_samples = []
        if sampling_probs is not None:            
            for i in true_classes:
                neg = np.random.choice(np.arange(len(self.label_list)),
                                       size=nb_samples,
                                       replace=False,
                                       p=sampling_probs[i])
                assert not np.isin(i, neg)
                neg = neg.tolist()
                neg_samples.append(neg)
        else:
            # Assume uniform distribution.
            for i in true_classes:
                choices = list(range(0,i)) + list(range(i+1, len(self.label_list)))
                neg = np.random.choice(choices, size=nb_samples, replace=False)
                neg = neg.tolist()
                neg_samples.append(neg)                

        # Overwrite self.data.
        pos_cand_ids = np.random.randint(0, high=(nb_samples+1), size=len(self.data), dtype=np.int)
        for i in range(len(self.data)):
            text = self.data[i][0]
            label_id = self.data[i][1]
            assert label_id == true_classes[i]
            # Insert true class at a random position among the neg samples            
            cands = neg_samples[i]
            pos_cand_id = pos_cand_ids[i]
            cands.insert(pos_cand_id, label_id)

            # Overwrite example
            self.data[i] = [text, label_id, cands, pos_cand_id]
        return

        
class DatasetForTestingClassifier(DatasetForClassification):
    """ A class for evaluating language identification (i.e. multi-class classification) on dev or test sets. """
    
    def __init__(self, path_data, tokenizer, seq_len, sort_by_length=False, require_labels=False, encoding="utf-8", verbose=False):
        """ Init

        Params:
        - path_data: path of a file in TSV format, with one or 2 columns, containing texts and optional labels.
        - tokenizer:
        - seq_len: maximum sequence length (including CLS)

        """
        self.tokenizer = tokenizer
        self.max_seq_len = seq_len
        self.verbose = verbose
        self.sort_by_length = sort_by_length
        self.label_list = sorted(ALL_LANGS)
        self.lang2id = {x:i for i,x in enumerate(self.label_list)}
        
        # Load data
        data = []
        with open(path_data, encoding=encoding) as f:
            for line in f:
                elems = line.strip().split("\t")
                if len(elems) == 0:
                    # Empty line
                    continue
                elif len(elems) == 1:
                    if require_labels:
                        msg = "only once column found, but require_labels is True"
                        raise RuntimeError(msg)
                    text = elems[0]
                    label_id = None
                    cands = None
                    pos_cand_id = None
                elif len(elems) == 2:
                    text = elems[0]
                    lang = elems[1]
                    label_id = self.lang2id[lang]
                    cands = [label_id]
                    pos_cand_id = 0
                else:
                    msg = "invalid number of columns (%d)" % len(elems)
                    raise RuntimeError(msg)
                data.append((text, label_id, cands, pos_cand_id))

        # Sort data by text length
        if self.sort_by_length:
            data = sort_examples_by_text_length(data, text_ix=0)
                
        # Init parent
        super().__init__(data)        
