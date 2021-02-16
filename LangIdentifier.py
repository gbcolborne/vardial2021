""" Module for language identification (i.e. single-label, multi-class classification) """

import numpy as np
import torch
import torch.nn as nn


class LangIdentifier(nn.Module):
    """ Language identifier. """
    
    def __init__(self, encoder, lang_list, detector_hidden_size=None, freeze_encoder=True):
        """ Constructor. 

        Params:
        - encoder: a BertForPretraining model
        - lang_list: list of languages handled (in order)

        """
        super().__init__()
        self.encoder = encoder
        self.lang_list = lang_list
        self.detector_input_size = self.encoder.config.hidden_size
        self.detector_hidden_size = detector_hidden_size
        self.nb_classes = len(self.lang_list)
        self.lang2id = {x:i for i,x in enumerate(self.lang_list)}        

        # Language identification head, which is basically a stack of
        # language detectors. Includes optional class-wise hidden
        # layers.
        self.has_detector_hidden_layer = self.detector_hidden_size is not None and self.detector_hidden_size > 0
        if not self.has_detector_hidden_layer:
            self.dense = nn.Linear(self.detector_input_size, self.nb_classes)
            nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('sigmoid'))            
        else:
            w1 = torch.zeros(self.detector_input_size,
                             self.detector_hidden_size,
                             self.nb_classes).float()
            self.w1 = nn.Parameter(w1)
            nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))            
            b1 = torch.zeros(self.detector_hidden_size,
                             self.nb_classes).float()
            self.b1 = nn.Parameter(b1)
            self.activation = nn.ReLU()
            w2 = torch.zeros(self.detector_hidden_size,
                             self.nb_classes).float()
            self.w2 = nn.Parameter(w2)
            nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('sigmoid'))            
            b2 = torch.zeros(self.nb_classes).float()
            self.b2 = nn.Parameter(b2)


        # Freeze encoder
        if freeze_encoder:
            self.freeze(self.encoder)
        else:
            self.unfreeze(self.encoder)


    def set_output_biases(self, biases):
        """ Set output logit biases.

        Args:
        - biases: 1-D numpy array of biases (floats)

        """
        assert type(biases) == np.ndarray
        assert np.abs(biases.sum() - 1) < 1e-6
        biases = torch.tensor(biases).to(self.w1.device if self.has_detector_hidden_layer else self.dense.device)
        if self.has_detector_hidden_layer:
            self.b2.data = biases
        else:
            self.dense.bias.data = biases
            

    def freeze(self, module):
        for p in module.parameters():
            p.requires_grad = False


    def unfreeze(self, module):
        for p in module.parameters():
            p.requires_grad = True
                        

    def forward(self, input_ids, input_mask, segment_ids, candidate_classes=None):
        """ Forward pass to logits.

        Params:
        - input_ids: input token IDs
        - input_mask: attention mask (1 for real tokens, 0 for padding)
        - segment_ids: token type (i.e. sequence) IDs
        - candidate_classes: (optional) subsampled class IDs (for training), of shape (batch size, nb candidates). 

        """
        encodings = self.encoder.encode_and_pool([input_ids, input_mask, segment_ids])
        batch_size = encodings.size()[0]
        if not self.has_detector_hidden_layer:
            logits = self.dense(encodings)
        else:
            z = torch.matmul(encodings, self.w1.view(self.detector_input_size, -1))
            z = z.view(batch_size, self.detector_hidden_size, self.nb_classes)
            z = z + self.b1
            a = self.activation(z)
            logits = torch.bmm(a.permute(2,0,1), self.w2.permute(1,0).unsqueeze(2))
            logits = logits.squeeze(2).permute(1,0)
            logits = logits + self.b2
        if candidate_classes is not None:
            nb_cands = candidate_classes.size()[1]
            rows = (torch.arange(batch_size).unsqueeze(1) * torch.ones((batch_size, nb_cands)).long()).view(-1)
            cols = candidate_classes.view(-1)
            logits = logits[rows, cols]
            logits = logits.view(batch_size, nb_cands)
        return logits


    
