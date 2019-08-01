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
import kaldi_io as kio
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', help='file with CMVN model')
parser.add_argument('out_file', help='output file where the model will be saved')
args = parser.parse_args()

if args.model is not None:
    # Load CMVN model
    in_file = open(args.model, 'rb')
    scaler = pickle.load(in_file)
    in_file.close()
else:
    scaler = StandardScaler()

# Fit the data
for utt, x in kio.read_mat_ark(sys.stdin.buffer):
    scaler.partial_fit(x)

# Dump CMVN model
out_file = open(args.out_file, 'wb')
pickle.dump(scaler, out_file)
out_file.close()



