#!/usr/bin/python3

# Copyright 2018 Peter Gazdik

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

import os
import argparse
from pfeats_model import PFeatsModel
from pfeats_generator import DataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('feats_tr')
parser.add_argument('feats_cv')
parser.add_argument('phones_tr')
parser.add_argument('phones_cv')
parser.add_argument('lang')
parser.add_argument('exp_dir')
parser.add_argument('base_model', nargs='?')
args = parser.parse_args()

phn_map = os.path.join(args.exp_dir, 'phn_sil_to_idx.int')
tmpdir_tr = os.path.join(args.exp_dir, 'tmpdir_tr')
tmpdir_cv = os.path.join(args.exp_dir, 'tmpdir_cv')
model_fname = os.path.join(args.exp_dir, 'final.h5')


tr_gen = DataGenerator(args.feats_tr, args.phones_tr, args.exp_dir,
                       phn_map, args.lang, tmpdir_tr, batch_size=256)
cv_gen = DataGenerator(args.feats_cv, args.phones_cv, args.exp_dir,
                       phn_map, args.lang, tmpdir_cv, batch_size=256)

if args.base_model:
    m = PFeatsModel.load(args.base_model)
else:
    m = PFeatsModel.build(tr_gen.input_dim(), tr_gen.output_dim())

# Initial training
m.train(tr_gen, cv_gen, model_fname, epochs=5, patience=5)
# Continue training until validation loss stagnates
m.train(tr_gen, cv_gen, model_fname, epochs=10, patience=0, min_delta=0.002)
# Scale learning rate by half after each epoch
m.train_decay(tr_gen, cv_gen, model_fname, epochs=30, patience=10)

tr_gen.delete_tmp_dir()
cv_gen.delete_tmp_dir()
