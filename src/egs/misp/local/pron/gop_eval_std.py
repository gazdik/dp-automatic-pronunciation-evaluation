#!/usr/bin/env python3

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
import helpers as hlp

# Parse arguments
parser = ArgumentParser()
parser.add_argument('feats')
parser.add_argument('phones_ali_force')
parser.add_argument('phones_per_frame_free')
parser.add_argument('phn_to_pdf')
parser.add_argument('flags')
parser.add_argument('eval_dir')
args = parser.parse_args()

score_fname = os.path.join(args.eval_dir, 'score.ark')
score_txt_fname = os.path.join(args.eval_dir, 'score.txt')
score_txt = open(score_txt_fname, 'w')

# Load kaldi files
feats_it = kio.read_mat_ark(args.feats)
seg_lengths_force_dict = {k:v for k, v in hlp.read_segment_lengths(args.phones_ali_force)}
phones_ali_force_dict = {k:v for k, v in hlp.read_phone_alignments(args.phones_ali_force)}
phones_pf_free_dict = {k:v for k, v in kio.read_vec_int_ark(args.phones_per_frame_free)}
flags_dict = {k:v for k, v in kio.read_vec_int_ark(args.flags)}
pdf_to_phn = hlp.Pdf2PhnFeats(args.phn_to_pdf)

min_score = float('inf')
max_score = float('-inf')

with open(score_fname, 'wb') as f:
    for utt, feats in feats_it:
        # Workaround to deal with missing alignments
        if utt not in seg_lengths_force_dict:
            continue

        flags = flags_dict[utt]
        feats_per_phone = pdf_to_phn.convert(feats)
        # print(utt, "feats_per_phone", feats_per_phone.shape)
        phones_pf_free = phones_pf_free_dict[utt]
        phones_ali_force = phones_ali_force_dict[utt]
        seg_lengths_force = seg_lengths_force_dict[utt]

        # Get only features for corresponding pdf states in the alignments
        feats_per_phone_free = hlp.np_pick(feats_per_phone, phones_pf_free)

        # Calculate indexes of segments
        seg_idx_force = hlp.get_seg_idx(seg_lengths_force)

        scores = np.zeros_like(flags, dtype=float)
        for i in range(len(flags)):
            # Average features based on force alignment
            feats_num = feats_per_phone[seg_idx_force[i]:seg_idx_force[i + 1]]
            # print(i, "feats_num", feats_num.shape)
            # print(i, "feats_num[:, phone]", feats_num[:, phones_ali_force[i]])
            num_val = np.average(feats_num[:, phones_ali_force[i]])
            # print(i, "num_val", num_val)

            # Average features based on free alignment
            feats_den = feats_per_phone_free[seg_idx_force[i]:seg_idx_force[i + 1]]
            # print(i, "feats_den", feats_den.shape)
            den_val = np.average(feats_den)
            # print(i, "den_val", den_val)

            # Calc score
            scores[i] = np.abs(num_val - den_val)

            # Mark silence and insertion errors with nan value
            if flags[i] > 2:
                scores[i] = float('nan')

        if np.nanmin(scores) < min_score:
            min_score = np.nanmin(scores)
        if np.nanmax(scores) > max_score:
            max_score = np.nanmax(scores)

        print(hlp.flt_vec_to_str(utt, scores), file=score_txt)
        kio.write_vec_flt(f, scores, key=utt)

print("min_score", min_score, "max_score", max_score)
score_txt.close()
