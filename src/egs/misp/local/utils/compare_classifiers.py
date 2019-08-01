#!/usr/bin/python3

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

import os
import math
import kaldi_io as kio
import numpy as np
from helpers import read_inv_phone_map
from argparse import ArgumentParser

F_CORRECT = 0
F_SUBST = 1
F_DELET = 2
F_INSERT = 3
F_SILENT = 4
F_SILENT_MISMATCH = 5
F_UNKNOWN = 6

ERROR_DESC = {F_CORRECT: 'correct', F_SUBST: 'subsitution',
              F_DELET: 'deletion', F_INSERT: 'insertion',
              F_SILENT: 'silence', F_SILENT_MISMATCH: 'silence mismatch',
              F_UNKNOWN: 'unknown'}

parser = ArgumentParser()
parser.add_argument('flags', help='mispronunciation flags file')
parser.add_argument('text', help='text file with transcripts in integer form')
parser.add_argument('phone_map', help='phone map')
parser.add_argument('eval_dirs', nargs='+', help='eval directories with scores')
args = parser.parse_args()

phone_map = read_inv_phone_map(args.phone_map)

correct_means = [[] for _ in range(len(phone_map))]
correct_stds = [[] for _ in range(len(phone_map))]
correct_ranges = [[] for _ in range(len(phone_map))]
misp_means = [[] for _ in range(len(phone_map))]
misp_stds = [[] for _ in range(len(phone_map))]
misp_ranges = [[] for _ in range(len(phone_map))]

flags_dict = {k:v for k,v in kio.read_vec_int_ark(args.flags)}

# Create dictionaries with scores
score_dicts = []
for eval_dir in args.eval_dirs:
    score_file = os.path.join(eval_dir, 'score.ark')
    score_dict = {k:v for k,v in kio.read_vec_flt_ark(score_file)}
    score_dicts.append(score_dict)

# Iterate through all the texts
for key, phones in kio.read_vec_int_ark(args.text):
    # Workaround: Skip unscored utterances
    if key not in score_dicts[0]:
        continue

    scores_list = []
    for score_dict in score_dicts:
        scores_list.append(score_dict[key])
    flags = flags_dict[key]

    for i in range(len(phones)):
        # Calculate mean and std
        values = []
        for scores in scores_list:
            values.append(scores[i])
        mean = np.mean(values)
        std = np.std(values)
        val_range = np.max(values) - np.min(values)

        if flags[i] == F_CORRECT:
            correct_means[phones[i]].append(mean)
            correct_stds[phones[i]].append(std)
            correct_ranges[phones[i]].append(val_range)
        if flags[i] == F_SUBST or flags[i] == F_DELET:
            misp_means[phones[i]].append(mean)
            misp_stds[phones[i]].append(std)
            misp_ranges[phones[i]].append(val_range)

# Calculate pooled mean and std
print("%+7s" % "", "%-43s" % "CORRECT", "%-43s" % "MISPRONOUNCED", "%-20s" % "TOTAL")
print("%-7s" % "PHONE", "%-10s" % "MEAN", "%-10s" % "STD",
      "%-10s" % "RANGE MEAN", "%-10s" % "RANGE STD",
      "%-10s" % "MEAN", "%-10s" % "STD",
      "%-10s" % "RANGE MEAN", "%-10s" % "RANGE STD",
      "%-10s" % "MEAN", "%-10s" % "STD",
      "%-10s" % "RANGE MEAN", "%-10s" % "RANGE STD")
for phn in range(len(phone_map)):
    correct_mean = correct_std = misp_mean = misp_std = float('nan')
    correct_range_mean = correct_range_std = misp_range_mean = misp_range_std = float('nan')
    if len(correct_means[phn]) != 0:
        correct_mean = np.average(correct_means[phn])
        correct_std = np.average(np.square(correct_stds[phn]))
        correct_range_mean = np.mean(correct_ranges[phn])
        correct_range_std = np.std(correct_ranges[phn])
    if len(misp_means[phn]) != 0:
        misp_mean = np.average(misp_means[phn])
        misp_std = np.average(np.square(misp_stds[phn]))
        misp_range_mean = np.mean(misp_ranges[phn])
        misp_range_std = np.std(misp_ranges[phn])

    if math.isnan(correct_mean) and math.isnan(misp_mean):
        continue

    total_mean = np.average(correct_means[phn] + misp_means[phn])
    total_std = np.average(np.square(correct_stds[phn] + misp_stds[phn]))
    total_range_mean = np.mean(correct_ranges[phn] + misp_ranges[phn])
    total_range_std = np.std(correct_ranges[phn] + misp_ranges[phn])


    print("%-7s" % phone_map[phn], "%-10.5f" % correct_mean, "%-10.5f" % correct_std,
          "%-10.5f" % correct_range_mean, "%-10.5f" % correct_range_std,
          "%-10.5f" % misp_mean, "%-10.5f" % misp_std,
          "%-10.5f" % misp_range_mean, "%-10.5f" % misp_range_std,
          "%-10.5f" % total_mean, "%-10.5f" % total_std,
          "%-10.5f" % misp_range_mean, "%-10.5f" % misp_range_std)
