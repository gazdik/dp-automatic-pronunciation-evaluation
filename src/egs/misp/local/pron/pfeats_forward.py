#!/usr/bin/python3

# Copyright 2019 Peter Gazdik

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

import sys
import kaldi_io as kio
import numpy as np
import argparse
from pfeats_model import PFeatsModel

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Keras neural network')
parser.add_argument('priors', nargs='?', help='Prior probabilities in CSV format')
parser.add_argument('--inv_log', help='output probabilities in range [0, 1]', action='store_true')
args = parser.parse_args()

if not args.model.endswith('.h5'):
    raise TypeError('Unsupported model type. Please use h5 format.')

# Load model
m = PFeatsModel.load(args.model)
p = 1.0

if args.priors:
    p = np.genfromtxt(args.priors, delimiter=',')
    p[p == 0] = 1e-5  # Deal with zero priors

# Read feature vectors from stdin and forward them through the nnet
for utt_id, feat_mat in kio.read_mat_ark(sys.stdin.buffer):
    out_mat = m.predict(feat_mat) / p
    out_mat[out_mat == 0] = 1e-5

    if not args.inv_log:
        out_mat = np.log(out_mat)
        out_mat[out_mat == -np.inf] = -100

    kio.write_mat(sys.stdout.buffer, out_mat, key=utt_id)

