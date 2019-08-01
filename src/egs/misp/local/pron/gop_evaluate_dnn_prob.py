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

import sys
import os
import kaldi_io as kio
import numpy as np
from argparse import ArgumentParser
import helpers as hlp

# Parse arguments
parser = ArgumentParser()
parser.add_argument('data_dir', help='data directory')
parser.add_argument('cbps_dir', help='CBPS directory')
parser.add_argument('res_dir', help='output directory')
args = parser.parse_args()

# Prepere files
os.makedirs(args.res_dir, exist_ok=True)
feats_fd = sys.stdin.buffer
flags_fname = os.path.join(args.data_dir, 'text_ext_flags')
ali_force_frame_fname = os.path.join(args.cbps_dir, 'force_ali_test/ali_frames.gz')
ali_force_fname = os.path.join(args.cbps_dir, 'force_ali_test/ali_pdf.gz')
score_fname = os.path.join(args.res_dir, 'score.ark')
score_txt_fname = os.path.join(args.res_dir, 'score.txt')
score_txt = open(score_txt_fname, 'w')
cmp_fd = open('test/cmp_gop.txt', 'w')

# Load kaldi files
flags_it = kio.read_vec_int_ark(flags_fname)
ali_force_it = kio.read_ali_ark(ali_force_fname)
ali_force_frm_it = kio.read_ali_ark(ali_force_frame_fname)
feats_it = kio.read_mat_ark(feats_fd)

with open(score_fname, 'wb') as f:
    for flags_t, ali_force_t, ali_force_frm_t, feats_t in zip(flags_it, ali_force_it, ali_force_frm_it, feats_it):
        # Unpack each tuple
        utt, flags = flags_t
        _, ali_force = ali_force_t
        _, ali_force_frm = ali_force_frm_t
        _, feats = feats_t

        # Get only features for corresponding states in alignments
        probs_force = hlp.np_pick(feats, ali_force)

        # Calculate indexes of segments
        seg_lengths = hlp.get_seg_lengths(ali_force_frm)
        seg_idx = hlp.get_seg_idx(seg_lengths)

        scores = np.zeros_like(flags, dtype=float)
        for i in range(len(flags)):
            # Get probabilities for the actual segment
            probs = probs_force[seg_idx[i]:seg_idx[i + 1]]
            avg_prob = np.average(probs)
            scores[i] = avg_prob

            # Mark silence and insertion errors with nan value
            if flags[i] > 2:
                scores[i] = float('nan')

        print(hlp.flt_vec_to_str(utt, scores), file=score_txt)
        kio.write_vec_flt(f, scores, key=utt)
