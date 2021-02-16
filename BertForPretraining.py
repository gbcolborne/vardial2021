import torch
from torch import nn
from transformers import BertForMaskedLM, BertConfig
from Pooler import Pooler
from CosineBertHead import CosineBertHead


class BertForPretraining(nn.Module):
    """Bert for pre-training with masked language modeling and,
    optionally, sentence pair classification.

    """

    def __init__(self, config, args):
        """ Constructor.

        Args:
        - config: a BertConfig
        - args
        
        """
        super().__init__()
        self.config = config
        self.tasks = args.tasks
        assert self.tasks in ['mlm-only', 'spc-dot', 'spc-cos']
        self.avg_pooling = (not args.use_cls_for_spc)
        
        # Initialize BertForMaskedLM
        self.bert_for_mlm = BertForMaskedLM(config)
        
        # Initialize pooler and head for SPC
        if self.tasks == 'mlm-only':
            self.pooler = None
            self.sim = None
            self.spc_head = None
        else:
            self.pooler = Pooler(self.bert_for_mlm.config.hidden_size,
                                 cls_only=(not self.avg_pooling))
            if self.tasks == 'spc-cos':
                self.sim = nn.CosineSimilarity()
                self.spc_head = CosineBertHead(enforce_w_positivity=True)
            else:
                self.sim = None
                self.spc_head = None

    def encode(self, inputs):
        """ Encode sequence and return last hidden states.

        Args:
        - inputs: list containing following tensors for a batch of sequences:
            - input token IDs
            - attention mask
            - segment IDs

        """
        input_ids = inputs[0]
        input_mask = inputs[1]
        segment_ids = inputs[2]        
        outputs = self.bert_for_mlm.bert(input_ids=input_ids,
                                         attention_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         position_ids=None)
        last_hidden_states = outputs[0] # Last hidden states, shape (batch_size, seq_len, hidden_size)
        return last_hidden_states


    def encode_and_pool(self, inputs):
        """ Encode sequence and pool (or take encoding of CLS token).

        Args:
        - inputs: list containing following tensors for a batch of sequences:
            - input token IDs
            - attention mask
            - segment IDs

        """
        last_hidden_states = self.encode(inputs)
        if self.pooler is None:
            encodings = last_hidden_states[:,0,:]
        else:
            encodings = self.pooler(last_hidden_states)
        return encodings
                
    def forward(self, query_inputs, cand_inputs=None):
        """ Forward pass.

        Args:
        - query_inputs: list containing following tensors for a batch of queries: 
            - input token IDs (masked for MLM)
            - attention mask
            - segment IDs
        - (optional) cand_inputs: list containing following tensors for a batch of candidates:
            - un-masked query input token IDs (like input token IDs of the query, but not masked for MLM.
            - input token IDs
            - attention mask
            - segment IDs

        """
        if cand_inputs is None:
            assert self.tasks == 'mlm-only'
            
        # Call BERT model to get encoding of query
        query_last_hidden_states = self.encode(query_inputs) # Last hidden states, shape (batch_size, seq_len, hidden_size)

        # Do MLM on last hidden states obtained using query inputs.
        mlm_pred_scores = self.bert_for_mlm.cls(query_last_hidden_states)
        all_outputs = []
        all_outputs.append(mlm_pred_scores)
        
        if self.tasks != 'mlm-only':
            assert cand_inputs is not None
            
            # Get encodings of query and candidates for SPC
            query_inputs = [cand_inputs[0], query_inputs[1], query_inputs[2]]
            cand_inputs = cand_inputs[1:]            
            query_last_hidden_states = self.encode(query_inputs)
            query_encodings = self.pooler(query_last_hidden_states)
            cand_last_hidden_states = self.encode(cand_inputs)
            cand_encodings = self.pooler(cand_last_hidden_states)

            # Score candidates
            if self.tasks == 'spc-dot':
                # Score candidates using dot(query, candidate)
                spc_scores = torch.bmm(query_encodings.unsqueeze(1), cand_encodings.unsqueeze(2)).squeeze(2).squeeze(1)
            else:
                # Score candidates using CosineBert
                sims = self.sim(query_encodings, cand_encodings)
                spc_scores = self.spc_head(sims, return_logits=True)
            all_outputs.append(spc_scores)
        return all_outputs
