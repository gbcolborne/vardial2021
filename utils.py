import os, math, logging, pickle
from iteround import saferound
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, IterableDataset
from comp_utils import ALL_LANGS, RELEVANT_LANGS, IRRELEVANT_LANGS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_mcm_to_accuracy(mcm):
    """Given a multilabel confusing matrix, return class-wise
    (one-vs-rest) accuracy.

    """
    # Make sure we have a valid mcm
    assert type(mcm) == np.ndarray
    assert len(mcm.shape) == 3
    assert mcm.shape[1] == 2
    assert mcm.shape[2] == 2

    # Compute class-wise (one-vs-rest) accuracy
    nb_correct = mcm[:,0,0] + mcm[:,1,1]
    total = np.sum(np.sum(mcm, 2), 1) # Should all be equal
    assert np.all(total == total[0])
    accuracies = nb_correct.astype(float) / total
    return accuracies
    

def compute_neg_sampling_probs(conf):
    """Given a confusion matrix, compute an array of class-wise
    negative sampling probs.

    """
    # Make sure we have a valid confusion matrix
    assert type(conf) == np.ndarray
    assert len(conf.shape) == 2
    assert conf.shape[0] == conf.shape[1]

    # Compute neg sampling probs
    nb_classes = conf.shape[0]
    probs = conf.astype(np.float)
    probs = probs + probs.T # Sum rows and columns
    probs += 1 # Laplace smoothing
    np.fill_diagonal(probs, 0)
    probs /= np.sum(probs,1)[:,np.newaxis] #Row-wise normalization
    return probs


def compute_sampling_probs(args, dev_scores=None):
    """Compute sampling probs. 

    Params:
    - args

    - dev_scores: 1-D numpy array of class-wise dev scores, required
      for importance sampling. These scores should be between 0 (bad)
      and 1 (good).

    """
    # Check args
    assert args.sampling_crit in ['uniform', 'frequency', 'importance-a', 'importance-ar']
    if args.sampling_crit == 'frequency':
        assert args.sampling_alpha is not None
        assert args.weight_relevant is not None
        assert args.dir_train_data is not None
    if args.sampling_crit in ['importance-a', 'importance-ar']:
        assert dev_scores is not None
        assert type(dev_scores) == np.ndarray
        assert len(dev_scores.shape) == 1
        assert np.all(dev_scores >= 0)
        assert np.all(dev_scores <= 1)        
        
    # Compute sampling probs
    if args.sampling_crit == 'uniform':
        nb_langs = len(ALL_LANGS)
        sampling_probs = np.ones(nb_langs, dtype=np.float)
        sampling_probs /= nb_langs
    elif args.sampling_crit == 'frequency':
        lang2freq = load_or_make_lang2freq(args.dir_train_data)
        sampling_probs = compute_split_sampling_probs(lang2freq,
                                                      sampling_alpha=args.sampling_alpha,
                                                      weight_relevant=args.weight_relevant)
    elif args.sampling_crit == 'importance-a':
        sampling_probs = (1-dev_scores)
        sampling_probs /= sampling_probs.sum()
    elif args.sampling_crit == 'importance-ar':
        ranks = len(dev_scores) - dev_scores.argsort().argsort()
        sampling_probs = ranks / ranks.sum()
    return sampling_probs

        
def compute_split_sampling_probs(lang2freq, sampling_alpha=1.0, weight_relevant=1.0):
    # Map languages to ids
    all_langs = sorted(ALL_LANGS)
    lang2id = {x:i for i,x in enumerate(all_langs)}

    # Compute the sampling probabilities of the relevant and
    # irrelevant languages independently.
    rel_langs = sorted(RELEVANT_LANGS)
    irr_langs = sorted(IRRELEVANT_LANGS)
    logger.info("Computing sampling probabilities for relevant languages...")
    rel_probs = compute_sampling_probs_for_subgroup(rel_langs, lang2freq, sampling_alpha)
    logger.info("Computing sampling probabilities for irrelevant languages...")
    irr_probs = compute_sampling_probs_for_subgroup(irr_langs, lang2freq, sampling_alpha)
    # Weight the distribution of relevant languages, then renormalize
    rel_probs = rel_probs * weight_relevant
    sum_of_both = rel_probs.sum() + irr_probs.sum()
    rel_probs = rel_probs / sum_of_both
    irr_probs = irr_probs / sum_of_both
    sample_probs = np.zeros(len(all_langs), dtype=np.float)
    for lang, prob in zip(rel_langs, rel_probs):
        lang_id = lang2id[lang]
        sample_probs[lang_id] = prob
    for lang, prob in zip(irr_langs, irr_probs):
        lang_id = lang2id[lang]
        sample_probs[lang_id] = prob
    logger.info("Stats on sampling probabilities:")
    logger.info("- Min prob (relevant): %f" % (min(rel_probs)))
    logger.info("- Mean prob (relevant): %f" % (np.mean(rel_probs)))        
    logger.info("- Max prob (relevant): %f" % (max(rel_probs)))
    logger.info("- Cumulative prob (relevant): %f" % (sum(rel_probs)))        
    logger.info("- Min prob (irrelevant): %f" % (min(irr_probs)))
    logger.info("- Mean prob (irrelevant): %f" % (np.mean(irr_probs)))        
    logger.info("- Max prob (irrelevant): %f" % (max(irr_probs)))
    logger.info("- Cumulative prob (irrelevant): %f" % (sum(irr_probs)))        
    return sample_probs


def compute_sampling_probs_for_subgroup(lang_list, lang2freq, alpha):
    counts = np.array([lang2freq[k] for k in lang_list], dtype=np.float)
    probs = counts / counts.sum()
    probs_damp = probs ** alpha
    probs = probs_damp / probs_damp.sum()
    return probs


def compute_sample_sizes(probs, max_sample_sizes, total_sample_size, scale_total=False):
    """Compute sample sizes given sampling probabilities and corresponding
    list of maximum sample sizes, as well as the total sample size.

    Args:
    - probs:
    - max_sample_sizes
    - total_sample_size
    - scale_total: Flag that determines what to do if any of the
      initial expected sample sizes are greater than the corresponding
      max sample sizes. If scale_total is True, then we scale down the
      total_sample_size such that no expected samples sizes are
      greater than the corresponding max sample sizes. Otherwise, we
      recursively set sample sizes to their maximum value if their
      current expectation is greater than that maximum value.

    """
    assert type(probs) == np.ndarray
    if not probs.dtype == float:
        probs = probs.astype(float)
    assert type(max_sample_sizes) == np.ndarray 
    assert len(probs.shape) == 1   
    assert probs.shape == max_sample_sizes.shape
    
    if scale_total:
        # Compute expected sample sizes
        ess = probs * total_sample_size

        # Find the max ratio expected sample size / max sample
        # size. If the max ratio is greater than 1, then scale down
        # total sample size.
        ratios = ess / max_sample_sizes
        max_ratio = max(ratios)
        if max_ratio > 1:
            new_total = int(total_sample_size / max_ratio)
            logger.warning("scaling total_sample_size down from %d to %d" % (total_sample_size, new_total))
            total_sample_size = new_total
            
            # Re-compute the expected sample sizes
            ess = probs * total_sample_size

        # Round sample sizes using a sum-safe algorithm
        ss = saferound(ess, 0, 'difference')
        ss = [int(x) for x in ss]
        assert sum(ss) == total_sample_size
        return ss
    
    # So scale_total is False. Make sure we can actually reach the
    # total sample size
    assert max_sample_sizes.sum() >= total_sample_size
    
    # Initialize sample sizes
    ss = np.zeros(probs.shape, dtype=np.int)

    # Initialize mask indicating where the sample sizes have NOT been set yet
    mask = np.ones(probs.shape, dtype=np.int)

    # Recursively figure out where we need to set the sample sizes to the maximum value
    nb_iter = 0
    max_iter = 3
    rtss = total_sample_size # Remaining total sample size
    while True:
        # Compute expected sample sizes
        to_set = np.where(mask == 1)
        ess = np.round(probs[to_set] * rtss, 0).astype(np.int)
        
        # Check where expected sample size >= maximum sample size
        where_ess_geq_max = np.where(ess >= max_sample_sizes[to_set])
        if len(where_ess_geq_max[0]) == 0:
            break
        to_set_to_max = to_set[0][where_ess_geq_max]
        
        # Where expected sample size >= max sample size, set sample size to max sample size
        ss[to_set_to_max] = max_sample_sizes[to_set_to_max]
        mask[to_set_to_max] = 0
        to_set = np.where(mask == 1)
        if mask.sum() == 0:
            break
        
        # Compute remaining total sample size
        rtss = total_sample_size - ss.sum()
        if rtss == 0:
            break
        assert rtss > 0

        # Normalize sampling probabilities where sample size has not been set yet
        assert max_sample_sizes[to_set].sum() > rtss
        probs[to_set] /= probs[to_set].sum()
        
        nb_iter += 1
        if nb_iter > max_iter:
            msg = "There were still expected_ss >= max_ss after %d iterations" % max_iter
            raise RuntimeError(msg)
    msg = "Nb recursion steps it took to figure out where the sample sizes must be set to their maximum value: "
    msg += str(nb_iter)
    logger.info(msg)
    if rtss == 0:
        return ss
    
    # Round remaining sample sizes up or down using a sum-safe rounding algorithm
    to_set = np.where(mask == 1)
    rtss = total_sample_size - ss.sum()
    ess = probs[to_set] * rtss
    ss[to_set] = saferound(ess, 0, 'difference')
    if not ss.sum() == total_sample_size:
        msg = "Expected sample sizes to sum to %d, but they actually sum to %d" % (total_sample_size, ss.sum())
        raise RuntimeError(msg)
    return ss
                              

def load_or_make_lang2freq(dir_data, include_lengths=False, encoding="utf-8"):
    """Load or make a map of languages to the number of lines in the file
    for that language. Optionally, also include a map of languages to
    a list of text lengths for each language (1 per line).

    """
    path_freq = os.path.join(dir_data, "lang2freq.pkl")
    path_lens = os.path.join(dir_data, "lang2lens.pkl")
    if os.path.exists(path_freq) and ((not include_lengths) or os.path.exists(path_lens)):
        logger.info("Loading %s" % path_freq)
        with open(path_freq, 'rb') as f:
            lang2freq = pickle.load(f)
        if include_lengths:
            logger.info("Loading %s" % path_lens)
            with open(path_lens, 'rb') as f:
                lang2lens = pickle.load(f)
    else:
        if include_lengths:
            logger.info("Mapping languages to the number of training examples and text lengths")
            lang2lens = {}
        else:
            logger.info("Mapping languages to the number of training examples")
        lang2freq = {k:0 for k in ALL_LANGS}
        for lang in sorted(ALL_LANGS):
            p = os.path.join(dir_data, "%s.train" % lang)
            logger.info("  Processing %s" % p)
            with open(p, 'r', encoding=encoding) as f:
                lengths = []
                for line in f:
                    text, _ = line_to_data(line, False, strict=True)
                    assert text is not None
                    lang2freq[lang] += 1
                    if include_lengths:
                        lengths.append(len(text))
            if include_lengths:
                lang2lens[lang] = lengths
        logger.info("Saving map to %s" % path_freq)
        with open(path_freq, 'wb') as f:
            pickle.dump(lang2freq, f)
        if include_lengths:
            logger.info("Saving map to %s" % path_lens)
            with open(path_lens, 'wb') as f:
                pickle.dump(lang2lens, f)
    if include_lengths:
        return lang2freq, lang2lens
    else:
        return lang2freq


def sort_examples_by_text_length(data, text_ix=0):
    """ Sort data by text length. """
    # Check if data is a list of strings, or a list of lists
    if type(data[0]) not in [list, tuple]:
        assert type(data[0]) == str
        return sorted(data, key=len, reverse=False)
    assert type(data[0][text_ix]) == str
    lengths = [len(x[text_ix]) for x in data]
    argsort = np.argsort(lengths).tolist()
    result = [data[i][:] for i in argsort]
    return result


def line_to_data(line, is_labeled, strict=False):
    """Takes a line from a dataset (labeled or unlabeled), and returns the
    text and label.

    """
    if is_labeled:
        elems = line.strip().split("\t")
        assert len(elems) == 2
        text = elems[0]
        label = elems[1]
    else:
        text = line.strip()
        label = None        
        # Make sure text is not labeled
        if strict:
            if len(text) > 3:
                assert text[-4] != "\t"
    return (text, label)


def shorten_seqs_in_tensor(tensor, cutoff, dim=1):
    """Shorten sequences in tensor. """
    nb_axes = len(tensor.size())
    if nb_axes == 2:
        if dim == 0:
            output = tensor[:cutoff,:]
        else:
            output = tensor[:,:cutoff]
    elif nb_axes == 3:
        if dim == 0:
            output = tensor[:cutoff,:,:]
        elif dim == 1:
            output = tensor[:,:cutoff,:]
        else:
            output = tensor[:,:,:cutoff]
    else:
        msg = "tensor must have 2 or 3 axes"
        raise NotImplementedError(msg)
    if not output.is_contiguous():
        output = output.contiguous()
    return output


def get_state_dict_from_checkpoint(checkpoint_data, model_name):
    name = model_name + "_state_dict"
    key = "best_" + name if "best_" + name in checkpoint_data else name
    logger.info("Got weights from '%s'" % key)
    return checkpoint_data[key]


def log_model_info(logger, model):
    logger.info("Encoder config: %s" % repr(model.encoder.config))
    logger.info("Model params:")
    for n,p in model.named_parameters():
        msg = "  %s" % n
        if not p.requires_grad:
            msg += " ***FROZEN***"
        logger.info(msg)
    logger.info("Total params in model: %d" % count_params(model))
    logger.info("  Nb params in encoder: %d" % count_params(model.encoder))    
    for k in ["detector", "classifier"]:
        if k in dir(model):
            logger.info("  Nb params in %s: %d" % (k, count_params(model.__getattr__(k))))


class RandomSequentialBatchLoader(object):
    """Dataloader in which batches are in a random order, but the examples
    within each batch are sequential (i.e. an contiguous batch of
    examples from the dataset).

    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Compute number of batches
        self.nb_batches = int(math.ceil(len(dataset) / batch_size))

        # Create random ordering of batch indices
        self.batch_order = np.arange(self.nb_batches)
        np.random.shuffle(self.batch_order)
    
    def __getitem__(self, item):
        start = self.batch_order[item] * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))
        items = [self.dataset[i] for i in range(start, end)]
        batch_size = len(items)
        nb_features = len(items[0])
        batch = []
        for ix in range(nb_features):
            features = [x[ix] for x in items]
            nb_axes = len(features[0].size())
            if nb_axes == 0:
                features = torch.tensor(features)
            elif nb_axes == 1:
                features = torch.stack(features, dim=0)
            else:
                msg = "Unexpected number of axes: %d" % nb_axes
                raise RuntimeError(msg)
            batch.append(features)
        return batch
        
    def __len__(self):
        return self.nb_batches
    
    
def get_module(model):
    """Return model's module attribute if it exits, otherwise return the
    model itself. This can be useful when using torch.nn.DataParallel,
    for instance.

    """
    return model.module if hasattr(model, 'module') else model


def get_dataloader(dataset, batch_size, local_rank):
    """ Get data loader. """
    if issubclass(type(dataset), IterableDataset):
        # Iterable datasets can not be shuffled
        return DataLoader(dataset, batch_size=batch_size)
    else:
        # This is a map-style Dataset. I'm pretty sure none of my
        # map-style Datasets needs shuffling for the moment. 
        shuffle = False
        if local_rank == -1:
            sampler = SequentialSampler(dataset)            
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def accuracy_simple(pred_labels, true_labels):
    """Compute accuracy of predicted labels with respect to true
    labels.

    """
    ypred = pred_labels.cpu().detach().numpy()
    assert len(ypred.shape) == 1
    ytrue = true_labels.cpu().detach().numpy()
    assert len(ytrue.shape) == 1
    assert ypred.shape == ytrue.shape
    nb_correct = np.sum(ypred == ytrue)
    accuracy = nb_correct / ytrue.shape[0]
    return accuracy


def accuracy(pred_scores, labels, ignore_label=None):
    """Compute accuracy of predicted scores of a multi-class classifier
    with respect to the true labels.

    """
    ytrue = labels.cpu().numpy()
    assert len(ytrue.shape) == 1
    ypred = pred_scores.detach().cpu().numpy()
    ypred = np.argmax(ypred, axis=1)
    assert len(ypred.shape) == 1
    assert ypred.shape == ytrue.shape
    if ignore_label is not None:
        keep = np.where(ytrue != ignore_label)
        ytrue = ytrue[keep]
        ypred = ypred[keep]
    nb_correct = np.sum(ypred == ytrue)
    accuracy = nb_correct / ytrue.shape[0]
    return accuracy


def count_params(model):
    """ Count params in model. """
    count = 0
    for p in model.parameters():
         count += torch.prod(torch.tensor(p.size())).item()
    return count


def weighted_avg(vals, weights):
    """ Compute weighted average. """
    vals = np.asarray(vals)
    weights = np.asarray(weights)
    assert len(vals.shape) == 1
    assert vals.shape == weights.shape
    probs = weights / weights.sum()
    weighted_sum = np.sum(vals * probs)    
    return weighted_sum


def adjust_loss(loss, args):
    """ Adapt loss for distributed training or gradient accumulation. """
    if args.nb_gpus > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
    if args.grad_accum_steps > 1:
        loss = loss / args.grad_accum_steps
    return loss


def check_for_unk_train_data(train_paths):
    """ Check for a file named `unk.train`, containing unlabeled data. """
    for path in train_paths:
        if os.path.split(path)[-1] == "unk.train":
            return path
    return None

