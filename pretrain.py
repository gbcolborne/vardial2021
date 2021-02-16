"""Pre-train BERT for language identification using masked language
modeling and sequence pair classification.

"""

import os, argparse, logging, pickle, glob, math
from io import open
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertConfig
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import trange
from CharTokenizer import CharTokenizer
from LIDataset import DatasetForPretraining, NO_MASK_LABEL
from BertForPretraining import BertForPretraining
from utils import adjust_loss, weighted_avg, count_params, accuracy, accuracy_simple, get_module, RandomSequentialBatchLoader, shorten_seqs_in_tensor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, tokenizer, optimizer, scheduler, dataset, args, checkpoint_data):
    """Pretrain model
    
    Args:
    - model: BertModelForPretraining
    - tokenizer: CharTokenizer
    - optimizer
    - scheduler
    - dataset: DatasetForPretraining
    - args
    - checkpoint_data: dict


    """
        
    # Write header in log
    header = "GlobalStep\tLossMLM\tAccuracyMLM"
    if args.tasks != 'mlm-only':
        header += "\tLossSPC\tAccuracySPC"
    header += "\tGradNorm\tWeightNorm"
    with open(args.train_log_path, "w") as f:
        f.write(header + "\n")

    # Make dataloader(s). 
    dataloader = RandomSequentialBatchLoader(dataset, args.train_batch_size)

    # Lists used to compute stats for logging
    real_batch_sizes = []
    query_mlm_losses = []
    query_mlm_accs = []
    spc_losses = []
    spc_accs = []
    grad_norms = []
    
    # Start training
    logger.info("***** Running training *****")
    current_opt_step = checkpoint_data["global_step"]
    nb_opt_steps_left = checkpoint_data["max_opt_steps"] - current_opt_step
    model.train()
    for opt_step in trange(nb_opt_steps_left, desc="Opt steps"):
        for sub_step in range(args.grad_accum_steps):
            # Get batch
            batch_ix = checkpoint_data["batch_ix"]
            batch = dataloader[batch_ix]

            # Unpack batch
            batch = tuple(t.to(args.device) for t in batch)
            input_ids = batch[0]
            input_ids_masked = batch[1]
            input_masks = batch[2]
            segment_ids = batch[3]
            seq_lens = batch[4]
            lm_label_ids = batch[5]
            spc_input_ids = batch[6]
            spc_input_masks = batch[7]
            spc_segment_ids = batch[8]
            spc_seq_lens = batch[9]
            spc_labels = batch[10]
            real_batch_sizes.append(len(input_ids))

            # Shorten sequences if possible to remove padding
            if args.tasks == "mlm-only":
                cutoff = max(seq_lens)
            else:
                cutoff = max(max(seq_lens), max(spc_seq_lens))                
            input_ids = shorten_seqs_in_tensor(input_ids, cutoff)
            input_ids_masked = shorten_seqs_in_tensor(input_ids_masked, cutoff)            
            input_masks = shorten_seqs_in_tensor(input_masks, cutoff)
            segment_ids = shorten_seqs_in_tensor(segment_ids, cutoff)
            lm_label_ids = shorten_seqs_in_tensor(lm_label_ids, cutoff)
            if args.tasks != "mlm-only":
                spc_input_ids = shorten_seqs_in_tensor(spc_input_ids, cutoff)
                spc_input_masks = shorten_seqs_in_tensor(spc_input_masks, cutoff)
                spc_segment_ids = shorten_seqs_in_tensor(spc_segment_ids, cutoff)

            # Forward pass
            query_inputs = [input_ids_masked, input_masks, segment_ids]
            if args.tasks == "mlm_only":
                cand_inputs = None
            else:
                cand_inputs = [input_ids, spc_input_ids, spc_input_masks, spc_segment_ids]
            outputs = model(query_inputs, cand_inputs=cand_inputs)
            mlm_pred_scores = outputs[0]
            if args.tasks != 'mlm-only':
                spc_scores = outputs[1]
                
            # Compute loss
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
            query_mlm_losses.append(loss.item())
            if args.tasks != 'mlm-only':
                spc_loss_fct = BCEWithLogitsLoss(reduction="mean")
                spc_loss = spc_loss_fct(spc_scores, spc_labels.float())
                loss = loss + spc_loss
                spc_losses.append(spc_loss.item())

            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()

            # Compute norm of gradient
            training_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    training_grad_norm += torch.norm(param.grad, p=2).item()
            grad_norms.append(training_grad_norm)

            # Compute accuracies
            query_mlm_acc = accuracy(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1), ignore_label=NO_MASK_LABEL)
            query_mlm_accs.append(query_mlm_acc)
            if args.tasks != 'mlm-only':
                spc_preds = torch.gt(torch.sigmoid(spc_scores), 0.5).int()
                spc_acc = accuracy_simple(spc_preds, spc_labels)
                spc_accs.append(spc_acc)                

            # Increment batch ix, check if we need to resample training examples
            checkpoint_data["batch_ix"] += 1
            if checkpoint_data["batch_ix"] >= len(dataloader):
                checkpoint_data["batch_ix"] = 0
                dataset.resample()
                dataloader = RandomSequentialBatchLoader(dataset, args.train_batch_size)
                
        # Do optimization step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        checkpoint_data["global_step"] += 1

        # Check if this is a logging/checkpointing step
        if checkpoint_data["global_step"] % args.log_every == 0:
            # Compute stats for this epoch
            last_grad_norm = grad_norms[-1]
            avg_mlm_loss = weighted_avg(query_mlm_losses, real_batch_sizes)        
            avg_mlm_acc = weighted_avg(query_mlm_accs, real_batch_sizes)
            if args.tasks != 'mlm-only':
                avg_spc_loss = weighted_avg(spc_losses, real_batch_sizes)
                avg_spc_acc = weighted_avg(spc_accs, real_batch_sizes)

            # Compute norm of model weights
            weight_norm = 0
            for param in model.parameters():
                weight_norm += torch.norm(param.data, p=2).item()
            
            # Write stats for this epoch in log
            log_data = []
            log_data.append(str(checkpoint_data["global_step"]))
            log_data.append("{:.5f}".format(avg_mlm_loss))
            log_data.append("{:.5f}".format(avg_mlm_acc))
            if args.tasks != 'mlm-only':
                log_data.append("{:.5f}".format(avg_spc_loss))
                log_data.append("{:.5f}".format(avg_spc_acc))
            log_data.append("{:.5f}".format(last_grad_norm))
            log_data.append("{:.5f}".format(weight_norm))        
            with open(args.train_log_path, "a") as f:
                f.write("\t".join(log_data)+"\n")

            # Save checkpoint
            model_to_save = get_module(model)
            checkpoint_data['model_state_dict'] = model_to_save.state_dict()
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()        
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_path = os.path.join(args.output_dir, "checkpoint.tar")
            torch.save(checkpoint_data, checkpoint_path)            

            # Save dataset
            dataset_path = os.path.join(args.output_dir, "dataset.pkl")
            with open(dataset_path, "wb") as f:
                pickle.dump(dataset, f)
            
            # Reset lists used to compute stats
            real_batch_sizes = []
            query_mlm_losses = []
            query_mlm_accs = []
            spc_losses = []
            spc_accs = []
            grad_norms = []
            
        # Check if we are done
        if checkpoint_data["global_step"] >= checkpoint_data["max_opt_steps"]:
            break
                

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model_or_config_file", 
                        default=None, 
                        type=str, 
                        required=True,
                        help=("Path of configuration file (if starting from scratch) or directory"
                              " containing checkpoint (if resuming) or directory containig a"
                              " pretrained model and tokenizer (if re-training)."))

    # Use for resuming from checkpoint
    parser.add_argument("--resume",
                        action='store_true',
                        help="Resume from checkpoint")
    
    # Required if not resuming
    parser.add_argument("--dir_train_data",
                        type=str,
                        help="Path of a directory containing training files (names must all match <lang>.train)")
    parser.add_argument("--path_vocab",
                        type=str,
                        help="Path of a 2-column TSV file containing the vocab of chars and their frequency.")
    parser.add_argument("--output_dir",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--tasks",
                        choices=["mlm-only", "spc-dot", "spc-cos"],
                        default=["mlm-only"],
                        help=("Tasks: SPC means we do sentence pair classification along with MLM, "
                              "using either a simple dot product, or the CosineBERT architecture."))
    parser.add_argument("--use_cls_for_spc",
                        action="store_true",
                        help=("Use encoding of CLS token to to SPC, rather than pooling all the encodings. "
                              "Note that in either case, the resulting vector passes through a square linear layer "
                              "and a relu before the classification layer."))
    parser.add_argument("--sampling_alpha",
                        type=float,
                        default=1.0,
                        help="Dampening factor for relative frequencies used to compute language sampling probabilities")
    parser.add_argument("--weight_relevant",
                        type=float,
                        default=1.0,
                        help="Relative sampling frequency of relevant languages wrt irrelevant languages")
    parser.add_argument("--nb_train_samples",
                        default=100000,
                        type=int,
                        help="Size of training set we sample at beginning of each epoch")
    parser.add_argument("--max_opt_steps",
                        default=1000000,
                        type=int,
                        help="Maximum number of optimization steps to perform.")
    parser.add_argument("--log_every",
                        default=500,
                        type=int,
                        help="Number of optimization steps between logging/checkpointing steps")
    parser.add_argument("--num_warmup_steps",
                        default=10000,
                        type=int,
                        help="Number of optimization steps to perform linear learning rate warmup for. ")
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help="Number of training steps (i.e. batches) to accumualte before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seq_len",
                        default=128,
                        type=int,
                        help="Length of input sequences. Shorter seqs are padded, longer ones are trucated")
    parser.add_argument("--min_freq",
                        default=1,
                        type=int,
                        help="Minimum character frequency. Characters whose frequency is under this threshold will be mapped to <UNK>")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--equal_betas",
                        action='store_true',
                        help="Use beta1=beta2=0.9 for AdamW optimizer.")
    parser.add_argument("--no_bias_correction",
                        action='store_true',
                        help="Do not correct bias in AdamW optimizer (to reproduce BERT behaviour exactly.")
    parser.add_argument("--num_gpus",
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
                        help="random seed for initialization")
    args = parser.parse_args()

    # These args are required if we are not resuming from checkpoint
    if not args.resume:
        assert args.dir_train_data is not None
        assert args.path_vocab is not None
        assert args.output_dir is not None
        
    # Check whether we are starting from scratch, resuming from a checkpoint, or retraining a pretrained model
    from_scratch = (not args.resume) and (not os.path.isdir(args.bert_model_or_config_file))
    retraining = (not args.resume) and (not from_scratch)
    
    # Load config. Load or create checkpoint data.
    if from_scratch:
        logger.info("***** Starting pretraining job from scratch *******")
        config = BertConfig.from_json_file(args.bert_model_or_config_file)
        checkpoint_data = {}
    elif retraining:
        logger.info("***** Starting pretraining job from pre-trained model *******")
        logger.info("Loading pretrained model...")
        model = BertModelForPretraining.from_pretrained(args.bert_model_or_config_file)
        config = model.config
        checkpoint_data = {}
    elif args.resume:
        logger.info("***** Resuming pretraining job *******")
        logger.info("Loading checkpoint...")
        checkpoint_path = os.path.join(args.bert_model_or_config_file, "checkpoint.tar")        
        checkpoint_data = torch.load(checkpoint_path)
        # Make sure we haven't already done the maximum number of optimization steps
        if checkpoint_data["global_step"] >= checkpoint_data["max_opt_steps"]:
            msg = "We have already done %d optimization steps." % checkpoint_data["global_step"]
            raise RuntimeError(msg)
        logger.info("Resuming from global step %d" % checkpoint_data["global_step"])
        # Replace args with initial args for this job, except for num_gpus, seed and model directory
        current_num_gpus = args.num_gpus
        current_seed = args.seed
        checkpoint_dir = args.bert_model_or_config_file
        args = deepcopy(checkpoint_data["initial_args"])
        args.num_gpus = current_num_gpus
        args.seed = current_seed
        args.bert_model_or_config_file = checkpoint_dir
        args.resume = True
        logger.info("Args (most have been reloaded from checkpoint): %s" % args)
        # Load config
        config_path = os.path.join(args.bert_model_or_config_file, "config.json")
        config = BertConfig.from_json_file(config_path)        

    # Check args
    assert args.sampling_alpha >= 0 and args.sampling_alpha <= 1
    assert args.weight_relevant > 0
    if args.grad_accum_steps < 1:
        raise ValueError("Invalid grad_accum_steps parameter: {}, should be >= 1".format(args.grad_accum_steps))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if (not args.resume) and len(os.listdir(args.output_dir)) > 0:
        msg = "Directory %s is not empty" % args.output_dir
        raise ValueError(msg)
    
    # Make or load tokenizer
    if args.resume or retraining:
        logger.info("Loading tokenizer...")
        tokenizer_path = os.path.join(args.bert_model_or_config_file, "tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    elif from_scratch:
        logger.info("Making tokenizer...")
        assert os.path.exists(args.path_vocab)
        tokenizer = CharTokenizer(args.path_vocab)
        if args.min_freq > 1:
            tokenizer.trim_vocab(args.min_freq)
        # Adapt vocab size in config
        config.vocab_size = len(tokenizer.vocab)

        # Save tokenizer
        fn = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(fn, "wb") as f:
            pickle.dump(tokenizer, f)
    logger.info("Size of vocab: {}".format(len(tokenizer.vocab)))

    # Copy config in output directory
    if not args.resume:
        config_path = os.path.join(args.output_dir, "config.json")
        config.to_json_file(config_path)
        
    # What GPUs do we use?
    if args.num_gpus == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.nb_gpus = torch.cuda.device_count()
        device_ids = None
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
        args.nb_gpus = args.num_gpus
        if args.nb_gpus > 1:
            device_ids = list(range(args.nb_gpus))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.nb_gpus = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} nb_gpus: {}, distributed training: {}".format(
        args.device, args.nb_gpus, bool(args.local_rank != -1)))
    
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.nb_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)
                    
    # Prepare model 
    if from_scratch or args.resume:
        model = BertForPretraining(config, args)
        if args.resume:
            model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(args.device)

    # Distributed or parallel?
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed training.")
        model = DDP(model)
    elif args.nb_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Log some info on the model
    logger.info("Model config: %s" % repr(model.config))
    logger.info("Nb params: %d" % count_params(model))

    # Get training data
    if args.resume:
        dataset_path = os.path.join(args.bert_model_or_config_file, "dataset.pkl")
        logger.info("Reloading dataset from %s" % dataset_path)        
        with open(dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
    else:
        logger.info("Preparing dataset using data from %s" % args.dir_train_data)
        include_spc = args.tasks != 'mlm-only'
        train_dataset = DatasetForPretraining(args.dir_train_data,
                                              tokenizer,
                                              args.seq_len,
                                              max_sample_size=args.nb_train_samples,
                                              include_spc=include_spc,
                                              sampling_alpha=args.sampling_alpha,
                                              weight_relevant=args.weight_relevant,
                                              encoding="utf-8",
                                              seed=args.seed,
                                              verbose=True)
        dataset_path = os.path.join(args.output_dir, "dataset.pkl")
        logger.info("Saving dataset to %s" % dataset_path)        
        with open(dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

    # Check length of dataset
    dataset_length = len(train_dataset)

    # Store optimization steps performed and maximum number of optimization steps 
    if not args.resume:
        checkpoint_data["global_step"] = 0
        checkpoint_data["batch_ix"] = 0
        checkpoint_data["max_opt_steps"] = args.max_opt_steps

    # Log some info before training
    logger.info("*** Training info: ***")
    logger.info("Max optimization steps: %d" % args.max_opt_steps)
    logger.info("Gradient accumulation steps: %d" % args.grad_accum_steps)
    logger.info("Max optimization steps: %d" % checkpoint_data["max_opt_steps"])
    if args.resume:
        logger.info("Nb optimization steps done so far: %d" % checkpoint_data["global_step"])
    logger.info("Total dataset size: %d examples" % (dataset_length))
    logger.info("Batch size: %d" % args.train_batch_size)
    logger.info("Log every: %d steps" % (args.log_every))
        
    # Prepare optimizer
    logger.info("Preparing optimizer...")
    np_list = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_params = [
        {'params': [p for n, p in np_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in np_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.equal_betas:
        betas = (0.9, 0.9)
    else:
        betas = (0.9, 0.999)
    optimizer = AdamW(opt_params,
                      lr=args.learning_rate,
                      betas=betas,
                      correct_bias=(not args.no_bias_correction)) # To reproduce BertAdam specific behaviour, use correct_bias=False
    if args.resume:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Prepare scheduler
    logger.info("Preparing learning rate scheduler...")
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=checkpoint_data["max_opt_steps"])
    if args.resume:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        logger.info("Current learning rate: %f" % scheduler.get_last_lr()[0])

    # Save initial training args
    if not args.resume:
        checkpoint_data["initial_args"] = args
    
    # Prepare training log
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_log_path = os.path.join(args.output_dir, "%s.train.log" % time_str)        
    args.train_log_path = train_log_path
    
    # Train
    train(model, tokenizer, optimizer, scheduler, train_dataset, args, checkpoint_data)


if __name__ == "__main__":
    main()
