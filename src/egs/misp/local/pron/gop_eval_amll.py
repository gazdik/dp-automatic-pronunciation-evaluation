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
parser.add_argument('flags')
parser.add_argument('phones_ali')
parser.add_argument('phn_to_pdf')
parser.add_argument('eval_dir')
args = parser.parse_args()

score_fname = os.path.join(args.eval_dir, 'score.ark')
score_txt_fname = os.path.join(args.eval_dir, 'score.txt')
score_txt = open(score_txt_fname, 'w')

# Load kaldi files
feats_it = kio.read_mat_ark(args.feats)
seg_lengths_dict = {k:v for k, v in hlp.read_segment_lengths(args.phones_ali)}
phone_ali_dict = {k:v for k, v in hlp.read_phone_alignments(args.phones_ali)}
flags_dict = {k:v for k, v in kio.read_vec_int_ark(args.flags)}
pdf_to_phn = hlp.Pdf2PhnFeats(args.phn_to_pdf)

out_binary = open(score_fname, 'wb')

with open(score_fname, 'wb') as f:
    for utt, feats in feats_it:
        # Workaround to deal with missing alignments
        if utt not in seg_lengths_dict:
            continue

        feats_per_phone = pdf_to_phn.convert(feats)
        flags = flags_dict[utt]
        phone_ali = phone_ali_dict[utt]
        seg_lengths = seg_lengths_dict[utt]

        # Calculate indexes of segments
        seg_idx = hlp.get_seg_idx(seg_lengths)

        scores = np.zeros_like(flags, dtype=float)
        for i in range(len(flags)):
            # Calc average likelihoods for current segment
            likelihoods_avg = np.average(feats_per_phone[seg_idx[i]:seg_idx[i + 1]], axis=0)

            # Calc result
            scores[i] = likelihoods_avg[phone_ali[i]]

            # Mark silence and insertion errors with nan value
            if flags[i] > 2:
                scores[i] = float('nan')

        print(hlp.flt_vec_to_str(utt, scores), file=score_txt)
        kio.write_vec_flt(f, scores, key=utt)

score_txt.close()
