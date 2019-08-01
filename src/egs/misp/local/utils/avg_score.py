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

import numpy as np
import kaldi_io as kio
from scipy.special import expit
from scipy.special import logit
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument('output', help='output file')
parser.add_argument('scores', nargs='+', help='score files')
args = parser.parse_args()

out_file = open(args.output, 'wb')

# Open all score files
scores_it = []
for score_fname in args.scores:
    x = kio.read_vec_flt_ark(score_fname)
    if next(x, None) is None:
        print("Invalid iterator", score_fname)
    scores_it.append(x)

while True:
    # Get scores for the next utterance
    scores = []
    for score_it in scores_it:
        try:
            key, values = next(score_it)
            scores.append(values)
        except StopIteration:
            print("Stop iterations")
            break
            s
    if not scores:
        break

    scores = np.array(scores)
    scores[scores == 1.0] = 0.99999
    scores[scores == 0.0] = 1e-8
    scores = logit(scores)

    # Average scores
    avg_scores = np.average(scores, axis=0)
    avg_scores = expit(avg_scores)

    kio.write_vec_flt(out_file, avg_scores, key=key)




