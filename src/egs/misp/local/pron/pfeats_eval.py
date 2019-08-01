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
from pfeats_map import PFeatsMap
from helpers import read_phone_map, read_inv_phone_map


parser = ArgumentParser()
parser.add_argument('pfeats', help='phonological features per frame')
parser.add_argument('align', help='alignments with phones per frame')
parser.add_argument('lang')
parser.add_argument('exp_dir', help='exp directory')
parser.add_argument('eval_dir')
args = parser.parse_args()

idx_to_phn_name_file = os.path.join(args.exp_dir, 'phn_sil_to_idx.txt')
phn_to_idx_file = os.path.join(args.exp_dir, 'phn_sil_to_idx.int')
pfeats_name_to_idx_file = os.path.join(args.exp_dir, 'pfeats_name_to_idx.txt')
out_file = os.path.join(args.eval_dir, 'accuracy.txt')
output = open(out_file, 'w')

# Mappings
idx_to_pfeats_name = read_inv_phone_map(pfeats_name_to_idx_file)
idx_to_phn_name = read_inv_phone_map(idx_to_phn_name_file)
pfeats_map = PFeatsMap(phn_to_idx_file, args.lang)

# Inputs
aligns_it = kio.read_vec_int_ark(args.align)
pfeats_it = kio.read_mat_ark(args.pfeats)

# Counters
phn_cnt = np.zeros(pfeats_map.phn_dim(), dtype=np.int)
phn_correct = np.zeros_like(phn_cnt)
pfeats_correct = np.zeros(pfeats_map.pfeats_dim(), dtype=np.int)

# Evaluate accuracy
for utt_phones, utt_pfeats_real in zip(aligns_it, pfeats_it):
    for phone, pfeats_real in zip(utt_phones[1], utt_pfeats_real[1]):
        pfeats_real = np.exp(pfeats_real)
        if not pfeats_map.is_phn_valid(phone):
            continue

        pfeats_true = pfeats_map.phn_to_pfeats(phone)
        pfeats_pred = np.round(pfeats_real)
        pfeats_flags = np.equal(pfeats_true, pfeats_pred)

        phn_cnt[pfeats_map.phn_to_idx(phone)] += 1
        pfeats_correct[pfeats_flags] += 1

        if np.all(pfeats_flags):
            phn_correct[pfeats_map.phn_to_idx(phone)] += 1

# Convert counts to percentage
phn_total = np.sum(phn_correct) / np.sum(phn_cnt)
phn_correct = phn_correct / phn_cnt
pfeats_correct = pfeats_correct / np.sum(phn_cnt)


# Print results
dash = '-' * 40
print(dash, file=output)
print('{:<6s}{:>16s}'.format('PHONE', 'ACCURACY'), file=output)
print(dash, file=output)
for phone, acc in enumerate(phn_correct):
    phn_name = idx_to_phn_name[phone]

    print('{:<6s}{:>16.3f}'.format(phn_name, acc), file=output)
print('{:<6s}{:>16.3f}'.format('Total', phn_total), file=output)

print(dash, file=output)
print('{:<6s}{:>16s}'.format('PFEAT', 'ACCURACY'), file=output)
print(dash, file=output)
for pfeat, acc in enumerate(pfeats_correct):
    pfeat_name = idx_to_pfeats_name[pfeat]
    print('{:<6s}{:>16.3f}'.format(pfeat_name, acc), file=output)

output.close()
