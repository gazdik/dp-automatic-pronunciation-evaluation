#!/usr/bin/env python3

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
import numpy as np
import kaldi_io as kio
import helpers as hlp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('alignments', help='phone alignments with segment lengths')
parser.add_argument('--inv_log', help='output probabilities in range [0, 1]', action='store_true')
parser.add_argument('--drop', type=int, default=0, help='number of frames that should be droped')
args = parser.parse_args()

# Read phone alignments into dictionary
seg_lengths = {k:v for k,v in hlp.read_segment_lengths(args.alignments)}

# Read state probabilities from stdin and calc 
# average log probabilities per segment
# for utt, probs in kio.read_mat_ark(sys.stdin.fileno()):
for utt, probs in kio.read_mat_ark(sys.stdin.buffer):
    # TODO sometimes aligning of canonical transcripts fails due to optimised L.fst???
    # Skip utterances with missing alignment
    if utt not in seg_lengths:
        print("avg_feats.py: Skipped utterance", utt, "(missing alignment)",
              file=sys.stderr)
        continue

    # Calculate lengths of segments and also cumulative sums for indexing
    seg_idx = np.cumsum(seg_lengths[utt])
    seg_idx = np.insert(seg_idx, 0, 0)

    # Calculate average log probabilities per segment
    output = np.zeros((len(seg_lengths[utt]), probs.shape[1]))
    for i in range(len(seg_lengths[utt])):
        weights = np.full_like(probs[seg_idx[i]:seg_idx[i+1]], 1.0)
        if args.drop > 0:
            weights[:args.drop] = 0
            weights[-args.drop:] = 0
        output[i] = np.average(probs[seg_idx[i]:seg_idx[i+1]], weights=weights, axis=0)

    if args.inv_log:
        output = np.exp(output)

    kio.write_mat(sys.stdout.buffer, output, key=utt)

