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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('pca_model', help='file with PCA model')
args = parser.parse_args()

# Load PCA model
in_file = open(args.pca_model, 'rb')
ipca = pickle.load(in_file)
in_file.close()

# Transform input data and write them to stdout
for utt, x in kio.read_mat_ark(sys.stdin.buffer):
    x_transformed = ipca.transform(x)
    kio.write_mat(sys.stdout.buffer, x_transformed, key=utt)
