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
args = parser.parse_args()

phone_map = read_inv_phone_map(args.phone_map)

categories_cnt = np.zeros(len(ERROR_DESC), dtype=int)
ins_cnt = np.zeros(len(phone_map), dtype=int)
sub_cnt = np.zeros(len(phone_map), dtype=int)
del_cnt = np.zeros(len(phone_map), dtype=int)
err_cnt = np.zeros(len(phone_map), dtype=int)
correct_cnt = np.zeros(len(phone_map), dtype=int)

flags_dict = {k:v for k,v in kio.read_vec_int_ark(args.flags)}
for key, phones in kio.read_vec_int_ark(args.text):
    flags = flags_dict[key]
    for phone, flag in zip(phones, flags):
        categories_cnt[flag] += 1
        if flag == F_CORRECT:
            correct_cnt[phone] += 1
        else:
            err_cnt[phone] += 1

        if flag == F_SUBST:
            sub_cnt[phone] += 1
        if flag == F_INSERT:
            ins_cnt[phone] += 1
        if flag == F_DELET:
            del_cnt[phone] += 1

total = np.sum(categories_cnt)
for i, val in enumerate(categories_cnt):
    print("%-20s" % (ERROR_DESC[i] + ':'), "%d" % val, "(%.2f %%)" % (val / total * 100))

err_total = np.sum(err_cnt)
correct_total = np.sum(correct_cnt)
for i, (err, sub, delet, ins, correct) in enumerate(zip(err_cnt, sub_cnt, del_cnt, ins_cnt, correct_cnt)):
    if err == 0 and correct == 0:
        continue
    print("%-10s" % (phone_map[i]), "Total: %7d" % (err + correct),
          "(%5.2f %%)" % ((err + correct) / (err_total + correct_total) * 100),
          "ERR: %7d" % err, "(%5.2f %%)" % (err / err_total * 100),
          "[SUB: %4d" % sub, "DEL: %4d" % delet, "INS: %4d]" % ins,
          "CORRECT: %7d" % correct, "(%5.2f %%)" % (correct / correct_total * 100))

