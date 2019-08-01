#!/usr/bin/python3

# Copyright 2018 Peter Gazdik
#           2016 D S Pavan Kumar <dspavankumar@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import kaldi_io as kio
import numpy as np
import argparse
from subprocess import Popen, PIPE


def output_feat_dim(exp):
    p = Popen (['am-info', exp+'/final.mdl'], stdout=PIPE)
    for line in p.stdout:
        if b'number of pdfs' in line:
            return int(line.split()[-1])


def compute_priors(feats_stream, exp_dir) -> np.array:
    """Compute prior probabilities of pdf states"""
    feats_dim = output_feat_dim(exp_dir)
    counts = np.zeros(feats_dim)

    # Count pdf states
    for utt, feats in kio.read_ali_ark(feats_stream):
        for feat in feats:
            counts[feat] += 1

    # Compute priors
    priors = counts / np.sum(counts)

    # Floor zero values
    priors[priors == 0] = 1e-5

    return priors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feats')
    parser.add_argument('exp_dir')
    parser.add_argument('out_file')
    args = parser.parse_args()

    priors = compute_priors(args.feats, args.exp_dir)

    # Write to file
    priors.tofile(args.out_file, sep=',', format='%e')



