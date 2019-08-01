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
import pickle
import numpy as np
import kaldi_io as kio
from sklearn.decomposition import IncrementalPCA
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('n_components', type=int, help='number of components to keep')
parser.add_argument('out_file', help='output file where the model will be saved')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size used for incremental PCA')
args = parser.parse_args()

batch_size = args.n_components * args.batch_size
ipca = IncrementalPCA(n_components=args.n_components, batch_size=batch_size)

# Fit PCA model with input data
batch_len = 0
batch = []
for utt, x in kio.read_mat_ark(sys.stdin.buffer):
    batch.append(x)
    batch_len += len(x)

    if batch_len > batch_size:
        ipca.partial_fit(np.vstack(batch))
        batch = []
        batch_len = 0

# Dump PCA model
out_file = open(args.out_file, 'wb')
pickle.dump(ipca, out_file)
out_file.close()



