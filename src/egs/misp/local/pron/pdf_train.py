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
from pdf_generator import DataGenerator
from pdf_model import PdfModel

parser = argparse.ArgumentParser()
parser.add_argument('feats_tr')
parser.add_argument('feats_cv')
parser.add_argument('pdf_tr')
parser.add_argument('pdf_cv')
parser.add_argument('exp_dir')
parser.add_argument('base_model', nargs='?')
parser.add_argument('--units_per_layer', default=512, type=int)
args = parser.parse_args()

tmpdir_tr = os.path.join(args.exp_dir, 'tmpdir_tr')
tmpdir_cv = os.path.join(args.exp_dir, 'tmpdir_cv')
model_fname = os.path.join(args.exp_dir, 'final.h5')

tr_gen = DataGenerator(args.feats_tr, args.pdf_tr, args.exp_dir,
                       tmpdir_tr, batch_size=256)
print("Training data loaded")
cv_gen = DataGenerator(args.feats_cv, args.pdf_cv, args.exp_dir,
                       tmpdir_cv, batch_size=256)
print("Validation data loaded")

if args.base_model:
    m = PdfModel.transfer_and_build(args.base_model,
                                    tr_gen.output_dim())
else:
    m = PdfModel.build(tr_gen.input_dim(), tr_gen.output_dim(),
                       units_per_layer=args.units_per_layer)

# Initial training
m.train(tr_gen, cv_gen, model_fname, epochs=5, patience=5)
# Continue training until validation loss stagnates
m.train(tr_gen, cv_gen, model_fname, epochs=15, patience=0, min_delta=0.002)
# Scale learning rate by half after each epoch
m.train_decay(tr_gen, cv_gen, model_fname, epochs=30, patience=5)

tr_gen.delete_tmp_dir()
cv_gen.delete_tmp_dir()
