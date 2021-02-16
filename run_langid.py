""" Train or evaluate language identifier. """

import sys, os, argparse, pickle, random, logging, math, glob
from io import open
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers import AdamW
from tqdm import tqdm, trange
from CharTokenizer import CharTokenizer
from LIDataset import DatasetForTrainingClassifier, DatasetForTestingClassifier
from LangIdentifier import LangIdentifier
from BertForPretraining import BertForPretraining
from utils import adjust_loss, weighted_avg, count_params, accuracy, get_dataloader, get_module
from utils import RandomSequentialBatchLoader, log_model_info, shorten_seqs_in_tensor
from utils import convert_mcm_to_accuracy, compute_sampling_probs, compute_neg_sampling_probs
from comp_utils import ALL_LANGS
from scorer import compute_fscores

DEBUG=False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, eval_dataset, args):
    """ Evaluate model on multi-class language identification. 

    Args:
    - model: LangIdentifier
    - eval_dataset: DatasetForTestingClassifier
    - args

    Returns: Dict containing scores

    """
    # Get logits (un-normalized class scores) from model. 
    dataloader = RandomSequentialBatchLoader(eval_dataset, args.eval_batch_size)
    logits = predict(model, dataloader, args)

    # Extract labels
    gold_labels = []
    for batch in dataloader:
        labels = batch[4].to(args.device)
        gold_labels.append(labels.unsqueeze(0))
    gold_labels = torch.cat(gold_labels, 1).squeeze(0)

    # Loss is cross-entropy
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(logits, gold_labels)
    scores = {}
    scores["loss"] = loss.item()
    
    # Compute f-scores
    pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()    
    gold_labels = gold_labels.detach().cpu().numpy().tolist()
    pred_labels = [eval_dataset.label_list[i] for i in pred_labels]
    gold_labels = [eval_dataset.label_list[i] for i in gold_labels]
    fscore_dict = compute_fscores(pred_labels, gold_labels, verbose=False)
    scores.update(fscore_dict)
    return scores


def predict(model, dataloader, args):
    """ Get predicted scores (un-normalized) for examples in dataset. 

    Args:
    - model: LangIdentifier
    - dataloader: a dataloader that wraps a DatasetForTestingClassifier
    - args


    Returns: predicted scores, tensor of shape (nb examples, nb classes)

    """

    scores = []
    model.eval()
    for step, batch in enumerate(tqdm(dataloader, desc="Prediction")):
        # Unpack batch
        batch = tuple(t.to(args.device) for t in batch)
        input_ids = batch[0]
        input_mask = batch[1]
        segment_ids = batch[2]
        seq_lens = batch[3]
        
        # Shorten sequences if possible to remove padding        
        cutoff = max(seq_lens)
        input_ids = shorten_seqs_in_tensor(input_ids, cutoff)
        input_mask = shorten_seqs_in_tensor(input_mask, cutoff)
        segment_ids = shorten_seqs_in_tensor(segment_ids, cutoff)

        # Do forward pass to get logits
        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids)
        scores.append(logits)
    scores_tensor = torch.cat(scores, dim=0)
    return scores_tensor


def train(model, optimizer, tokenizer, args, checkpoint_data, dev_dataset=None):
    """ Train model. 

    Args:
    - model: LangIdentifier
    - optimizer
    - tokenizer
    - args
    - checkpoint_data: dict
    - dev_dataset: (optional) DatasetForTestingClassifier

    Returns: None

    """
    if args.eval_during_training:
        assert dev_dataset is not None
        assert type(dev_dataset) == DatasetForTestingClassifier
    nb_classes = len(ALL_LANGS)
    
    # Where do we save stuff?
    save_to_dir = args.dir_classifier if args.resume else args.dir_output

    # Compute sampling probabilities
    if args.sampling_crit in ['importance-a', 'importance-ar']:
        dev_scores = np.zeros(nb_classes, dtype=np.float)
        sampling_probs = compute_sampling_probs(args, dev_scores=dev_scores)
    elif args.sampling_crit == "uniform":
        sampling_probs = None
    else:
        sampling_probs = compute_sampling_probs(args, dev_scores=None)

    # Make dataset
    logger.info("Making training set using data in %s..." % (args.dir_train_data))
    train_dataset = DatasetForTrainingClassifier(args.dir_train_data,
                                                 tokenizer,
                                                 sampling_probs,
                                                 sample_size=args.nb_train_samples,
                                                 max_seq_len=args.max_seq_len,
                                                 encoding="utf-8",
                                                 seed=(args.seed + checkpoint_data["global_step"]),
                                                 verbose=DEBUG)

    # Make dataloader
    train_dataloader = RandomSequentialBatchLoader(train_dataset, args.train_batch_size)
    nb_train_batches = len(train_dataloader)

    # Compute number of training and optimization steps we need to do
    nb_train_steps = nb_train_batches * args.nb_train_epochs    
    nb_opt_steps = nb_train_steps // args.grad_accum_steps
    
    # Compute set of milestones, i.e. optimization steps after which
    # we will do a logging/validation step
    milestones = set()
    current_opt_step = 0
    logger.info("Computing milestones")
    for e in range(args.nb_train_epochs):
        nb_opt_steps_in_epoch = int(((e+1) / args.nb_train_epochs) * nb_opt_steps) - int((e / args.nb_train_epochs) * nb_opt_steps) 
        current_opt_step += nb_opt_steps_in_epoch
        milestones.add(current_opt_step)

    # Log some info before we start
    logger.info("*** Training info: ***")
    logger.info("  Train dataset size: %d" % len(train_dataset))
    logger.info("  Train batch size: %d" % args.train_batch_size)
    logger.info("  Nb training batches in dataloader: %d" % nb_train_batches)
    logger.info("  Nb training epochs to do: %d" % args.nb_train_epochs)
    logger.info("  Nb training steps to do: %d" % nb_train_steps)
    logger.info("  Gradient accumulation steps: %d" % args.grad_accum_steps)    
    logger.info("  Nb optimization steps to do: %d" % nb_opt_steps)
    if args.eval_during_training:
        logger.info("  Validation dataset size: %d" % len(dev_dataset))
        logger.info("  Validation batch size: %d" % args.eval_batch_size)
    if args.resume:
        logger.info("  Resuming training")
        logger.info("    Optimization steps done: %d" % checkpoint_data["global_step"])
        logger.info("    Training steps done: %d" % checkpoint_data["train_step"])
    logger.info("Milestones: %s" % ", ".join([str(x) for x in sorted(milestones)]))
    
    # Prepare logs
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_log_name = "%s.%strain.log" % (time_str, "resume." if args.resume else "")
    train_log_path = os.path.join(save_to_dir, train_log_name)        
    header = "GlobalStep\tTrainLoss\tTrainAcc"
    header += "\tGradNorm\tWeightNorm"
    if args.eval_during_training:
        header += "\tDevLoss\tDevF1Track1\tDevF1Track2\tDevF1Track3"
    with open(train_log_path, "w") as f:
        f.write(header + "\n")
    if args.eval_during_training and args.sampling_crit in ['importance-a', 'importance-ar']:
        extra_log_name = "%s.dev.scores.by.class.log" % time_str
        extra_log_path = os.path.join(save_to_dir, extra_log_name)
        header = "GlobalStep\t" + "\t".join(sorted(ALL_LANGS))
        with open(extra_log_path, 'w') as f:
            f.write(header + "\n")
            
    # Evaluate model on dev set before training
    if not args.resume:
        checkpoint_data["best_score_track1"] = 0
        checkpoint_data["best_score_track2"] = 0
        checkpoint_data["best_score_track3"] = 0        
    if args.eval_during_training:
        logger.info("Evaluating model on dev set before we start training...")        
        log_data = []
        log_data.append(str(checkpoint_data["global_step"]))
        log_data += [""] * 4
        dev_scores = evaluate(model, dev_dataset, args)
        for k in ["track1", "track2", "track3"]:
            if dev_scores[k] > checkpoint_data["best_score_%s" % k]:
                checkpoint_data["best_score_%s" % k] = dev_scores[k]
        log_data.append("{:.5f}".format(dev_scores["loss"]))
        log_data.append("{:.5f}".format(dev_scores["track1"]))
        log_data.append("{:.5f}".format(dev_scores["track2"]))
        log_data.append("{:.5f}".format(dev_scores["track3"]))                            
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data) + "\n")

    # Subsample negative classes
    if args.nb_neg_samples is not None and args.nb_neg_samples > 0:
        if args.neg_sampling_crit == 'confusion':
            conf = dev_scores["confusion_matrix"]
            neg_sampling_probs = compute_neg_sampling_probs(conf)
        else:
            neg_sampling_probs = None
        train_dataset.subsample_neg_classes(args.nb_neg_samples, sampling_probs=neg_sampling_probs)
    
    # Initialize some lists to compute stats at each validation/logging step
    real_batch_sizes = []
    train_losses = []
    train_nb_correct = []
    grad_norms = []

    # Train
    logger.info("***** Running training *****")
    remaining_opt_steps = nb_opt_steps - checkpoint_data["global_step"]
    for opt_step in trange(remaining_opt_steps, desc="Optimization step", leave=True):
        model.train()
        for ts in range(args.grad_accum_steps):
            # Get batch, run forward and backward passes
            batch_ix = checkpoint_data["train_step"] % nb_train_batches
            checkpoint_data["train_step"] += 1
            batch = train_dataloader[batch_ix]
            batch = tuple(t.to(args.device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            segment_ids = batch[2]
            seq_lens = batch[3]            
            labels = batch[4]
            cands = None if not args.nb_neg_samples else batch[5]
            pos_cand_ids = batch[6]
            real_batch_sizes.append(len(input_ids))

            # Shorten sequences if possible to remove padding
            cutoff = max(seq_lens)
            input_ids = shorten_seqs_in_tensor(input_ids, cutoff)
            input_mask = shorten_seqs_in_tensor(input_mask, cutoff)
            segment_ids = shorten_seqs_in_tensor(segment_ids, cutoff)

            # Forward pass to logits
            logits = model(input_ids, input_mask, segment_ids, candidate_classes=cands)
            # Compute loss
            loss_fct = CrossEntropyLoss(reduction="mean")            
            targets = labels if not args.nb_neg_samples else pos_cand_ids
            loss = loss_fct(logits, targets)
            train_losses.append(loss.item())
            
            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()
            
            # Compute norm of gradient
            training_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    training_grad_norm += torch.norm(param.grad, p=2).item()
            grad_norms.append(training_grad_norm)

            # Check how many predicted labels are correct
            pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            targets = targets.detach().cpu().numpy()
            nb_correct = (pred_labels == targets).sum()
            train_nb_correct.append(nb_correct)

        # Do optimization step
        optimizer.step()
        optimizer.zero_grad()
        checkpoint_data["global_step"] += 1

        # Check if this optimization step was a milestone
        is_milestone = checkpoint_data["global_step"] in milestones
        if is_milestone:
                
            # Compute stats for this epoch
            last_grad_norm = grad_norms[-1]
            avg_train_loss = weighted_avg(train_losses, real_batch_sizes)
            train_acc = sum(train_nb_correct) / sum(real_batch_sizes)

            # Compute norm of model weights
            weight_norm = 0
            for param in model.parameters():
                weight_norm += torch.norm(param.data, p=2).item()
                
            # Evaluate model on dev set
            if args.eval_during_training:
                dev_scores = evaluate(model, dev_dataset, args)
                if args.sampling_crit in ['importance-a', 'importance-ar']:
                    classwise_dev_scores = convert_mcm_to_accuracy(dev_scores["multilabel_confusion_matrix"])
                
            # Write stats 
            log_data = []
            log_data.append(str(checkpoint_data["global_step"]))
            log_data.append("{:.5f}".format(avg_train_loss))
            log_data.append("{:.5f}".format(train_acc))
            log_data.append("{:.5f}".format(last_grad_norm))
            log_data.append("{:.5f}".format(weight_norm))
            if args.eval_during_training:
                log_data.append("{:.5f}".format(dev_scores["loss"]))
                log_data.append("{:.5f}".format(dev_scores["track1"]))
                log_data.append("{:.5f}".format(dev_scores["track2"]))
                log_data.append("{:.5f}".format(dev_scores["track3"]))                            
            with open(train_log_path, "a") as f:
                f.write("\t".join(log_data)+"\n")
            if args.eval_during_training and args.sampling_crit in ['importance-a', 'importance-ar']:
                with open(extra_log_path, 'a') as f:
                    log_data = [str(checkpoint_data["global_step"])]
                    log_data += ["%.4f" % x for x in classwise_dev_scores.tolist()]
                    f.write("\t".join(log_data)+"\n")
                    
            # Save best models if score has improved
            if args.eval_during_training:

                for k in ["track1", "track2", "track3"]:
                    if dev_scores[k] > checkpoint_data["best_score_%s" % k]:
                        checkpoint_data["best_score_%s" % k] = dev_scores[k]
                        model_to_save = get_module(model)
                        checkpoint_data['best_model_state_dict_%s' % k] = deepcopy(model_to_save.state_dict())

            # Save checkpoint
            model_to_save = get_module(model)
            checkpoint_data['model_state_dict'] = model_to_save.state_dict()
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint_path = os.path.join(save_to_dir, "checkpoint.tar")
            torch.save(checkpoint_data, checkpoint_path)

            # Check if we are done
            if checkpoint_data["global_step"] >= nb_opt_steps:
                break

            # Sample new training set for next epoch
            if args.resample_every_epoch:
                if args.sampling_crit in ['importance-a', 'importance-ar']:
                    sampling_probs = compute_sampling_probs(args, dev_scores=classwise_dev_scores)
                elif args.sampling_crit == "uniform":
                    sampling_probs = None
                else:
                    sampling_probs = compute_sampling_probs(args, dev_scores=None)
                train_dataset.resample(sampling_probs)
                # Subsample negative classes
                if args.nb_neg_samples is not None and args.nb_neg_samples > 0:
                    if args.neg_sampling_crit == 'confusion':
                        conf = dev_scores["confusion_matrix"]
                        neg_sampling_probs = compute_neg_sampling_probs(conf)
                    else:
                        neg_sampling_probs = None
                    train_dataset.subsample_neg_classes(args.nb_neg_samples, sampling_probs=neg_sampling_probs)
                train_dataloader = RandomSequentialBatchLoader(train_dataset, args.train_batch_size)
                assert len(train_dataloader) == nb_train_batches
                
            # Re-initialize lists used to compute stats
            real_batch_sizes = []
            train_losses = []
            train_nb_correct = []
            grad_norms = []
            
    logger.info("Done training language identifier.")
    return None
        
def main():
    parser = argparse.ArgumentParser()

    # Paths for model
    parser.add_argument("--dir_pretrained_model",
                        type=str,
                        help=("Dir containing pre-trained model from which we load the encoder. "
                              "If --do_train, either this or --dir_classifier must be specified."))
    parser.add_argument("--dir_classifier",
                        type=str,
                        help=("Dir containing pre-trained classifier (language identifier). "
                              "Required if --resume, --do_pred or --do_eval. If --do_train, "
                              "either this or --dir_pretrained_model must be specified."))
    parser.add_argument("--track",
                        choices=["1","2","3"],
                        default="3",
                        help=("Track to use if loading pretrained classifier weights from checkpoint. "
                              "If checkpoint does not contain weights optimized for that track, we simply load "
                              "'model_state_dict'."))
    parser.add_argument("--dir_output",
                        type=str,
                        help=("Directory in which model or results will be written. "
                              "Required if --do_train or --do_pred."))

    # Data
    parser.add_argument("--dir_train_data",
                        type=str,
                        help=("Dir containing training data (n files named <lang>.train containing unlabeled text). "
                              "Required if --do_train."))
    parser.add_argument("--path_dev",
                        type=str,
                        help=("Path of 2-column TSV file containing labeled validation examples. "
                              "Required if --do_eval."))
    parser.add_argument("--path_test",
                        type=str,
                        help=("Path of text file containing unlabeled test examples. "
                              "Required if --do_pred."))
    
    # Execution modes
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume training model in --dir_classifier (note: --dir_output will be ignored)")
    parser.add_argument("--do_train",
                        action="store_true",
                        help=("Create and train model using either a pretrained encoder (--dir_pretrained_model) "
                              "or a previously fine-tuned classifier (--dir_classifier)."))
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Evaluate model on dev set")
    parser.add_argument("--do_pred",
                        action="store_true",
                        help="Run prediction on test set")
    
    # Hyperparameters
    parser.add_argument("--detector_hidden_size",
                        type=int,
                        help="Size of hidden size of detectors (optional).")
    parser.add_argument("--unfreeze_encoder",
                        action="store_true",
                        help="Unfreeze weights of pre-trained encoder.")
    parser.add_argument("--sampling_crit",
                        choices=['uniform', 'frequency', 'importance-a', 'importance-ar'],
                        help=("Criterion used for sampling training examples (required if do_train). "
                              "If this is 'frequency', you can also specify sampling_alpha and weight_relevant"))
    parser.add_argument("--nb_train_samples",
                        type=int,
                        help="Number of training examples sampled for training.")
    parser.add_argument("--resample_every_epoch",
                        action="store_true",
                        help="Sample a new training set every epoch.")
    parser.add_argument("--sampling_alpha",
                        type=float,
                        default=1.0,
                        help="Dampening factor for relative frequencies used to compute language sampling probabilities")
    parser.add_argument("--weight_relevant",
                        type=float,
                        default=1.0,
                        help="Relative sampling frequency of relevant languages wrt irrelevant languages")
    parser.add_argument("--nb_neg_samples",
                        type=int,
                        default=None,
                        help="Nb negative classes to evaluate for each observation (if None or 0, we evaluate them all)")
    parser.add_argument("--neg_sampling_crit",
                        choices=["random", "confusion"],
                        help="Criterion used for subsampling of negative classes")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--max_seq_len",
                        default=256,
                        type=int,
                        help="Length of input sequences. Shorter seqs are padded, longer ones are trucated")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--equal_betas",
                        action='store_true',
                        help="Use beta1=beta2=0.9 for AdamW optimizer.")
    parser.add_argument("--no_bias_correction",
                        action='store_true',
                        help="Do not correct bias in AdamW optimizer (to reproduce BERT behaviour).")
    parser.add_argument("--nb_train_epochs",
                        default=3,
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help="Number of training steps (i.e. batches) to accumualte before performing a backward/update pass.")
    parser.add_argument("--nb_gpus",
                        type=int,
                        default=-1,
                        help="Num GPUs to use for training (0 for none, -1 for all available)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed")
    args = parser.parse_args()

    # Distributed or parallel?
    if args.local_rank != -1 or args.nb_gpus > 1:
        raise NotImplementedError("No distributed or parallel training available at the moment.")
    if torch.cuda.is_available():
        # For now we don't allow multi-gpus
        if args.nb_gpus > 1:
            raise NotImplementedError("Multi-gpu not implemented yet")
        args.device = torch.device("cuda")
        args.nb_gpus = 1
    else:
        args.device = torch.device("cpu")
        args.nb_gpus = 0
    
    # Check execution mode and other args
    assert args.resume or args.do_train or args.do_eval or args.do_pred
    if args.do_train:
        assert not args.resume
        assert not args.do_eval
        assert not args.do_pred
        assert args.sampling_crit is not None
        args.eval_during_training = args.path_dev is not None        
        if args.sampling_crit in ['importance-a', 'importance-ar']:
            assert args.path_dev is not None        
        assert args.dir_pretrained_model is not None or args.dir_classifier is not None
        assert not (args.dir_pretrained_model is not None and args.dir_classifier is not None)
        args.from_scratch = args.dir_pretrained_model is not None
        args.dir_init_model = args.dir_pretrained_model if args.from_scratch else args.dir_classifier
        assert args.dir_train_data is not None
        assert args.nb_train_samples is not None
        assert args.dir_output is not None        

    if args.resume:
        assert not args.do_eval
        assert not args.do_pred        
        assert args.dir_classifier is not None
    if args.do_eval:
        assert args.dir_classifier is not None
        assert args.path_dev is not None
    if args.do_pred:
        assert args.dir_classifier is not None
        assert args.path_test is not None
        assert args.dir_output is not None
    if args.dir_output is not None:
        if os.path.exists(args.dir_output):
            if os.path.isdir(args.dir_output) and len(os.listdir(args.dir_output)) > 1:
                msg = "%s already exists and is not empty" % args.dir_output
                raise ValueError(msg)
        else:
            os.makedirs(args.dir_output)
    if args.nb_neg_samples is not None:
        assert args.nb_neg_samples >= 0
        if args.nb_neg_samples > 0:
            assert args.do_train
            assert args.nb_neg_samples < len(ALL_LANGS)
            assert args.neg_sampling_crit is not None
    if args.grad_accum_steps < 1:
        raise ValueError("Invalid grad_accum_steps parameter: {}, should be >= 1".format(
                            args.grad_accum_steps))

    # Load or initialize checkpoint(s) to build or evaluate classifier
    if args.resume or args.do_eval or args.do_pred:
        checkpoint_path = os.path.join(args.dir_classifier, "checkpoint.tar")
        logger.info("Loading classifier checkpoint from %s" % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
    if args.resume:
        # Replace args with initial args for this job, except for
        # those we need to keep
        current_nb_gpus = args.nb_gpus
        current_dir_classifier = args.dir_classifier
        args = deepcopy(checkpoint_data["initial_args"])
        args.nb_gpus = current_nb_gpus
        args.dir_classifier = current_dir_classifier
        args.resume = True
        logger.info("Args (most have been reloaded from checkpoint): %s" % args)
    if args.do_train and (not args.resume):
        init_checkpoint_path = os.path.join(args.dir_init_model, "checkpoint.tar")
        logger.info("Loading checkpoint from %s" % init_checkpoint_path)
        init_checkpoint_data = torch.load(init_checkpoint_path)
        checkpoint_data = {}
        checkpoint_data["global_step"] = 0
        checkpoint_data["train_step"] = 0
        checkpoint_data["initial_args"] = args
        
    # Copy some args we need if we are just evaluating
    if args.do_eval or args.do_pred:
        args.max_seq_len = checkpoint_data["initial_args"].max_seq_len
        args.seed = checkpoint_data["initial_args"].seed
        args.detector_hidden_size = checkpoint_data["initial_args"].detector_hidden_size

    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.nb_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_dir = args.dir_init_model if args.do_train and (not args.resume) else args.dir_classifier
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Get validation data or test data
    dev_dataset = None
    if args.do_eval or (args.do_train and args.eval_during_training):
        logger.info("Loading validation data from %s..." % args.path_dev)
        dev_dataset = DatasetForTestingClassifier(args.path_dev,
                                                  tokenizer,
                                                  args.max_seq_len,
                                                  sort_by_length=True,
                                                  require_labels=True,
                                                  encoding="utf-8",
                                                  verbose=False)
    if args.do_pred:
        logger.info("Loading test data from %s..." % args.path_test)                                
        test_dataset = DatasetForTestingClassifier(args.path_test,
                                                   tokenizer,
                                                   args.max_seq_len,
                                                   sort_by_length=False,
                                                   require_labels=False,
                                                   encoding="utf-8",
                                                   verbose=DEBUG)

    # Initialize encoder from config
    logger.info("Initializing encoder")
    config_dir = args.dir_init_model if args.do_train and (not args.resume) else args.dir_classifier
    path = os.path.join(config_dir, "config.json")
    encoder_config = BertConfig.from_json_file(path)
    encoder_args = init_checkpoint_data["initial_args"] if args.do_train and (not args.resume) else checkpoint_data["initial_args"]
    # Hack to make sure we have the necessary args
    if 'tasks' not in encoder_args:
        encoder_args.tasks = 'mlm-only'
    if 'use_cls_for_spc' not in encoder_args:
        encoder_args.use_cls_for_spc = False
    encoder = BertForPretraining(encoder_config, encoder_args)

    # Initialize model
    logger.info("Initializing model")
    lang_list = sorted(ALL_LANGS)
    freeze = not args.unfreeze_encoder
    model = LangIdentifier(encoder,
                           lang_list,
                           detector_hidden_size=args.detector_hidden_size,
                           freeze_encoder=freeze)
    model.to(args.device)

    # Load pre-trained weights
    if args.do_train and (not args.resume):
        if args.from_scratch:
            logger.info("Loading pre-trained encoder weights")
            encoder_state_dict = init_checkpoint_data["model_state_dict"]
            model.encoder.load_state_dict(encoder_state_dict)
        else:
            logger.info("Loading pre-trained classifier weights")
            k = "best_model_state_dict_track%s" % args.track
            if k in init_checkpoint_data:
                logger.info("Loading model weights from '%s'" % k) 
                model.load_state_dict(init_checkpoint_data[k])
            else:
                logger.info("Loading model weights from 'model_state_dict'")             
                model.load_state_dict(init_checkpoint_data["model_state_dict"])            
    else:
        logger.info("Loading pre-trained classifier weights")
        k = "best_model_state_dict_track%s" % args.track        
        if k in checkpoint_data and not args.resume:
            logger.info("Loading model weights from '%s'" % k) 
            model.load_state_dict(checkpoint_data[k])
        else:
            logger.info("Loading model weights from 'model_state_dict'")             
            model.load_state_dict(checkpoint_data["model_state_dict"])            

    # Log some info on the model
    log_model_info(logger, model)

    # Save tokenizer and config
    if args.do_train and (not args.resume):
        path_config = os.path.join(args.dir_output, "config.json")
        model.encoder.config.to_json_file(path_config)
        output_path = os.path.join(args.dir_output, "tokenizer.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(tokenizer, f)
    
    # Train
    if args.do_train or args.resume:
        # Prepare optimizer
        logger.info("Preparing optimizer")
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.equal_betas:
            betas = (0.9, 0.9)
        else:
            betas = (0.9, 0.999)
        optimizer = AdamW(opt_params,
                          lr=args.learning_rate,
                          betas=betas,
                          correct_bias=(not args.no_bias_correction)) 
                
        # Load optimizer state if resuming
        if args.resume:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        # Run training 
        train(model,
              optimizer,
              tokenizer,
              args,
              checkpoint_data,
              dev_dataset=dev_dataset)

        # Reload checkpoint and model
        save_to_dir = args.dir_classifier if args.resume else args.dir_output
        checkpoint_data = torch.load(os.path.join(save_to_dir, "checkpoint.tar"))
        k = "best_model_state_dict_track%s" % args.track
        if k in checkpoint_data:
            model.load_state_dict(checkpoint_data[k])
        else:
            model.load_state_dict(checkpoint_data["model_state_dict"])

    # Evaluate model on dev set
    if args.do_eval:
        logger.info("*** Running evaluation... ***")
        scores = evaluate(model, dev_dataset, args)
        logger.info("***** Evaluation Results *****")
        for score_name in ["track1", "track2", "track3"]:
            logger.info("- %s: %.4f" % (score_name, scores[score_name]))

    # Get model's predictions on test set
    if args.do_pred:
        logger.info("*** Running prediction... ***")
        dataloader = get_dataloader(test_dataset, args.eval_batch_size, args.local_rank)
        logits = predict(model, dataloader, args)
        pred_class_ids = np.argmax(logits.cpu().numpy(), axis=1)
        pred_labels = [test_dataset.label_list[i] for i in pred_class_ids]
        path_pred = os.path.join(args.dir_output, "pred.txt")
        logger.info("Writing predictions in %s..." % path_pred)
        with open(path_pred, 'w', encoding="utf-8") as f:
            for x in pred_labels:
                f.write("%s\n" % x)
        
if __name__ == "__main__":
    main()
