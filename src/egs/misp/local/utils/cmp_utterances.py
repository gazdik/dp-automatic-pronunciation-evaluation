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

import helpers as hlp
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument('text_1', help='the first text file (in integer form)')
parser.add_argument('text_2', help='the second text file (in integer form)')
args = parser.parse_args()

# Open both files
# if args.text_1.endswith(".int"):

iter1 = hlp.read_text_file(args.text_1)
iter2 = hlp.read_text_file(args.text_2)

print("Comparing text files", args.text_1, "and", args.text_1)
err_cnt = 0
for x, y in zip(iter1, iter2):
    if x[0] != y[0]:
        err_cnt += 1
        if err_cnt > 42:
            raise Exception("Files contains different keys or aren't sorted.")
        continue

    if not hlp.utterance_equal(x[1], y[1]):
        print("Utterances", x[0], "differs")
        print(" ", hlp.utterance2str(x[1]))
        print(" ", hlp.utterance2str(y[1]))

