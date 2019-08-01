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
import kaldi_io as kio
import numpy as np
from argparse import ArgumentParser
from helpers import read_phone_map, get_segment_idxs
from keras.preprocessing.sequence import pad_sequences
from misp_model import MispModel

parser = ArgumentParser()
parser.add_argument('feats')
parser.add_argument('phones')
parser.add_argument('flags')
parser.add_argument('exp_dir')
parser.add_argument('eval_dir')
args = parser.parse_args()

os.makedirs(args.eval_dir, exist_ok=True)
model_fname = os.path.join(args.exp_dir, 'final.h5')
phn_map_fname = os.path.join(args.exp_dir, 'phn_sil_to_idx.int')
score_fname = os.path.join(args.eval_dir, 'score.ark')
score_txt_fname = os.path.join(args.eval_dir, 'score.txt')
score_txt = open(score_txt_fname, 'w')

# Load phone map
phn_map = read_phone_map(phn_map_fname)

# Load kaldi files
flags_dict = {k:v for k, v in kio.read_vec_int_ark(args.flags)}
ali_dict = {k:v for k, v in kio.read_ali_ark(args.phones)}
feats_it = kio.read_mat_ark(args.feats)

# Load classifier model
model = MispModel.load(model_fname)

with open(score_fname, 'wb') as f:
    for utt, feats in feats_it:
        # Workaround to deal with missing alignments
        if utt not in ali_dict:
            continue

        print(utt, end=' ', file=score_txt)
        flags = flags_dict[utt]
        ali = ali_dict[utt]
        scores = np.zeros_like(flags, dtype=float)

        # Split frames into segments
        segment_idxs = get_segment_idxs(ali)
        feats_segments = np.split(feats, segment_idxs)
        feats_segments = pad_sequences(feats_segments)
        ali_segments = np.split(ali, segment_idxs)

        predicts = model.predict(feats_segments)
        for i in range(len(ali_segments)):
            phone = ali_segments[i][0]
            flag = flags[i]
            predict = predicts[i]

            # Mark silence and insertion errors with NaN value
            if flag > 2:
                scores[i] = float('nan')
            else:
                scores[i] = predict[phn_map[phone]]

            print("%.2f" % scores[i], end=' ', file=score_txt)

        print("", file=score_txt)
        kio.write_vec_flt(f, scores, key=utt)

score_txt.close()