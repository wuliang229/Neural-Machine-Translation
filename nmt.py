# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 64]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 512]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np

import torch
from torch import nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # Encoder
        self.encoder_embed = nn.Embedding(len(vocab.src), embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size // 2, 1, batch_first = True, bidirectional = True)

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab.tgt), embed_size)
        self.decoder = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first = True)

        # Final output
        self.Ws = nn.Linear(2 * hidden_size, len(vocab.tgt))
        self.loss = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'sum')
        self.dropout = nn.Dropout(dropout_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)


    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_encodings, decoder_init_state = self.encode(src_sents)
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)

        return scores

    def encode(self, src_sents: List[List[str]]) -> Tuple[torch.Tensor, Tuple]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in other formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        # 1. one-hot
        one_hot = self.vocab.src.words2indices(src_sents)
        lengths = torch.tensor([len(s) for s in one_hot]).to(self.device)
        padded_one_hot_tensor = rnn.pad_sequence([torch.tensor(s) for s in one_hot], batch_first = True).to(self.device) # (batch_size, seq_len)

        # 2. Embedding 
        embedding = self.encoder_embed(padded_one_hot_tensor) # (batch_size, seq_len, embed_size)

        # 3. LSTM
        packed_embedding = rnn.pack_padded_sequence(embedding, lengths, batch_first = True, enforce_sorted = False)       
        src_encodings, decoder_init_state = self.encoder(packed_embedding)
 
        h = decoder_init_state[0].transpose(0, 1).reshape(1, len(lengths), -1) # (1, batch_size, hidden_size)
        c = decoder_init_state[1].transpose(0, 1).reshape(1, len(lengths), -1)

        del lengths
        del padded_one_hot_tensor
        del embedding
        del packed_embedding
        del decoder_init_state

        return src_encodings, (h, c)

    def decode(self, src_encodings: torch.Tensor, decoder_init_state: Tuple, tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        # 1. one-hot
        one_hot = self.vocab.tgt.words2indices(tgt_sents)
        lengths = torch.tensor([len(s) - 1 for s in one_hot]).to(self.device)
        padded_one_hot_tensor = rnn.pad_sequence([torch.tensor(s[:-1]) for s in one_hot], batch_first = True).to(self.device) # (batch_size, seq_len)

        # 2. Unpack src_encodings
        padded_src_encodings, src_lengths = rnn.pad_packed_sequence(src_encodings, batch_first = True) # (batch_size, seq_len, hidden_size)

        # 3. Embedding 
        embedding = self.decoder_embed(padded_one_hot_tensor) # (batch_size, seq_len, embed_size)

        # 4. Decode step by step with attention
        current_decoder_state = decoder_init_state
        entire_output = None

        current_embedding = torch.zeros(embedding.size(0), 1, padded_src_encodings.size(2)).to(self.device)

        for i in range(embedding.size(1)):

            current_embedding = torch.cat((current_embedding, embedding[:, i:i+1, :]), dim = 2) # (batch, embed + hidden)

            # 4.1 LSTM
            lstm_output, current_decoder_state = self.decoder(current_embedding, current_decoder_state) # (batch_size, 1, hidden_size)

            # 4.2 Attention
            alignment_vector = torch.bmm(padded_src_encodings, lstm_output.transpose(1, 2)).view(len(lengths), -1) # (batch_size, hidden_size)

            mask = torch.arange(alignment_vector.size(1)) < src_lengths.unsqueeze(1)

            alignment_vector = mask.float().to(self.device) * torch.softmax(alignment_vector, dim = 1)

            masked_attention = alignment_vector / torch.sum(alignment_vector, dim = 1).unsqueeze(1) # (batch, seq_len)

            weighted_average = torch.bmm(padded_src_encodings.transpose(1, 2), masked_attention.unsqueeze(2)).transpose(1, 2) # (batch, 1, hidden)
 
            concatenated_output = self.dropout(torch.cat((weighted_average, lstm_output), dim = 2)) # (batch, 1, hidden * 2)

            current_embedding = weighted_average

            # Accumulate outputs
            if entire_output is None:
                entire_output = concatenated_output
            else:
                entire_output = torch.cat((entire_output, concatenated_output), dim = 1)

        # 5. Final layer
        logits = self.Ws(entire_output) # (batch_size, seq_len, tgt_vocab_size)

        # 6. Calculate loss
        target = rnn.pad_sequence([torch.tensor(s[1:]) for s in one_hot]).to(self.device) # (batch_size, seq_len)

        scores = self.loss(logits.transpose(1, 2), target.transpose(0, 1)) 

        del lengths
        del padded_one_hot_tensor
        del padded_src_encodings
        del src_lengths
        del embedding
        del current_decoder_state
        del entire_output
        del current_embedding
        del lstm_output
        del alignment_vector
        del mask
        del masked_attention
        del weighted_average
        del concatenated_output
        del logits
        del target

        return scores

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_encodings, decoder_init_state = self.encode([src_sent])
        current_word = torch.ones(1, 1).long().to(self.device) # start of sentence <s>

        # 2. Unpack src_encodings
        padded_src_encodings, src_lengths = rnn.pad_packed_sequence(src_encodings, batch_first = True) # (batch_size, seq_len, hidden_size)

        current_decoder_state = decoder_init_state
        entire_output = None

        current_embedding = torch.zeros(1, 1, padded_src_encodings.size(2)).to(self.device)

        result = [[]]
        for i in range(max_decoding_time_step):

            embedding = self.decoder_embed(current_word) # (batch_size, 1, embed_size)

            current_embedding = torch.cat((current_embedding, embedding), dim = 2) # (batch, embed + hidden)

            lstm_output, current_decoder_state = self.decoder(current_embedding, current_decoder_state) # (batch_size, 1, hidden_size)

            alignment_vector = torch.bmm(padded_src_encodings, lstm_output.transpose(1, 2)).view(1, -1) # (batch_size, hidden_size)

            mask = torch.arange(alignment_vector.size(1)) < src_lengths.unsqueeze(1)

            alignment_vector = mask.float().to(self.device) * torch.softmax(alignment_vector, dim = 1)

            masked_attention = alignment_vector / torch.sum(alignment_vector, dim = 1).unsqueeze(1) # (batch, seq_len)

            weighted_average = torch.bmm(padded_src_encodings.transpose(1, 2), masked_attention.unsqueeze(2)).transpose(1, 2) # (batch, 1, hidden)
 
            concatenated_output = self.dropout(torch.cat((weighted_average, lstm_output), dim = 2)) # (batch, 1, hidden * 2)

            current_embedding = weighted_average

            logits = self.Ws(concatenated_output)

            current_word = torch.argmax(logits, dim = 2)

            if current_word[0,0] == 2: # End of sentence </s>
                break

            result[0].append(self.vocab.tgt.id2word[current_word[0,0].item()]) 

        del src_encodings
        del decoder_init_state
        del padded_src_encodings
        del src_lengths
        del embedding
        del current_decoder_state
        del entire_output
        del current_embedding
        del lstm_output
        del alignment_vector
        del mask
        del masked_attention
        del weighted_average
        del concatenated_output
        del logits
        del current_word

        return result

    def evaluate_ppl(self, dev_data: List[List[str]], batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self(src_sents, tgt_sents)

                cum_loss += loss
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = math.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src') # List of tokenized sentences
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)

    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            model.train()
            optimizer.zero_grad()

            train_iter += 1

            batch_size = len(src_sents)

            # (batch_size)
            loss = model(src_sents, tgt_sents)
            loss.backward()
            optimizer.step()

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         math.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples))

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                model.eval()
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!')
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr)

                        # load model
                        model = NMT.load(model_save_path)

                        print('restore parameters of the optimizers')
                        # You may also need to load the state of the optimizer saved before
                        # TODO

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!')
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    # if args['TEST_TARGET_FILE']:
    #     test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}")
    model = NMT.load(args['MODEL_PATH'])
    model.eval()

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    # if args['TEST_TARGET_FILE']:
    #     top_hypotheses = [hyps[0] for hyps in hypotheses] # Find the sentence with the highest likelihood
    #     bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    #     print(f'Corpus BLEU: {bleu_score}')

    with open(args['OUTPUT_FILE'], 'w') as f:
        for hyps in hypotheses:
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')

def init_weights(m):
    """
    helper function that initializes the parameters uniformly between -0.1 and 0.1
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
