import argparse
import torch
import codecs
import os
import math
import json

from itertools import count

from onmt.models.model import NMTModel
from onmt.encoders import RNNEncoder, MeanEncoder
import onmt.opts


# def make_encoder(opt, out_file=None):
#
#     if out_file is None:
#         out_file = codecs.open(opt.output, 'w', 'utf-8')
#
#     if opt.gpu > -1:
#         torch.cuda.set_device(opt.gpu)
#
#     dummy_parser = argparse.ArgumentParser(description='train.py')
#     onmt.opts.model_opts(dummy_parser)
#     dummy_opt = dummy_parser.parse_known_args([])[0]
#     dummy_opt = dummy_opt.__dict__
#
#     checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
#     fields = onmt.io.load_fields_from_vocab(checkpoint['vocab'], data_type=opt.data_type)
#
#     model_opt = checkpoint['opt']
#     for arg in dummy_opt:
#         if arg not in model_opt:
#             model_opt.__dict__[arg] = dummy_opt[arg]
#
#     src_dict = fields["src"].vocab
#     feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
#     src_embeddings = onmt.ModelConstructor.make_embeddings(model_opt, src_dict,
#                                      feature_dicts)
#
#     encoder = RNNEncoder(model_opt.rnn_type, model_opt.brnn, model_opt.enc_layers,
#                          model_opt.rnn_size, model_opt.dropout, src_embeddings,
#                          model_opt.bridge)
#
#     return encoder


def make_iOS_files(opt):

    result_dir = 'iOS_files_' + opt.model

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    dummy_opt = dummy_opt.__dict__

    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    with open(result_dir + '/model_params.txt', 'w') as outfile:
        json.dump(model_opt.__dict__, outfile, ensure_ascii=False, indent=2, sort_keys=True)

    src_dict = checkpoint['vocab'][0][1]
    with open(result_dir + '/src_vocab.json', 'w') as outfile:
        json.dump(src_dict.itos, outfile)

    tgt_dict = checkpoint['vocab'][1][1]
    with open(result_dir + '/tgt_vocab.json', 'w') as outfile:
        json.dump(tgt_dict.itos, outfile)

    for layer, weights in checkpoint['model'].items():
        with open(result_dir + ('/%s' % (layer,)), 'wb') as weights_file:
            weights_file.write(weights.numpy().tobytes())

    for layer, weights in checkpoint['generator'].items():
        with open(result_dir + ('/generator_%s' % (layer,)), 'wb') as weights_file:
            weights_file.write(weights.numpy().tobytes())

    return

