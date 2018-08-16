#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.VivienUtils import make_iOS_files
import onmt.opts


def main(opt):

    make_iOS_files(opt)

    # encoder = make_encoder(opt)


    # dummy_input = torch.ones([20, 1, 1], dtype=torch.int64)
    # torch.onnx.export(encoder.embeddings, dummy_input, 'encoderModel.proto', verbose=True)
    # model = onnx.load('encoderModel.proto')
    # coreml_model = convert(
    #     model,
    #     'encoder',
    #     # image_input_names=['input'],
    #     # image_output_names=['output'],
    #     # class_labels=[i for i in range(100)],
    # )
    # coreml_model.save('encoderModel.mlmodel')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    # logger = get_logger(opt.log_file)
    main(opt)
