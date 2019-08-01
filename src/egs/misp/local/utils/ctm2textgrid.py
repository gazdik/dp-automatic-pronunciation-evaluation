#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Copyright 2017    Ming Tu

# This script contains the main function to convert ctm files to textgrid format files.
# This code is adapted from corresponding code in Montreal-Forced-aligner
# (https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git)

import argparse
from textgrid_ops import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ctm_fname', help='ctm file name')
    parser.add_argument('output_directory',help='output directory for textgrid files')
    parser.add_argument('phone_mapping',help='phone mapping')
    parser.add_argument('utt2dur', help = 'duration of each utterance')

    args = parser.parse_args()

    phone_ctm = {}
    parsed = parse_ctm(args.ctm_fname, args, mode='phone')
    for k, v in parsed.items():
        if k not in phone_ctm:
            phone_ctm[k] = v
        else:
            phone_ctm[k].update(v)

    ctm_to_textgrid(phone_ctm, args.output_directory, args.utt2dur)
