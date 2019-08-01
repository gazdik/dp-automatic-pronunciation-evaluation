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
from textgrid import TextGrid
import argparse
import copy


def add_scores(scores, textgrid_file):
    """
    Add score into textgrid file
    :param gop_vals: gop values of one utterance extracted from gop files
    :param textgrid_file: textgrid file
    :return:
    """

    textgrid = TextGrid()
    textgrid.read(textgrid_file)

    score_tier = copy.deepcopy(textgrid.tiers[0])
    textgrid.append(score_tier)

    for idx in range(len(scores)):
        if score_tier[idx].mark is not None:
            score_tier[idx].mark = "%0.2f" % scores[idx]

    textgrid.write(textgrid_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', help='file with scores in text format')
    parser.add_argument('ali_dir', help='directory with alignments in textgrid format')

    args = parser.parse_args()

    with open(args.scores, 'r') as scores_fd:
        for eachline in scores_fd:
            uttid = eachline.split()[0]
            score_str = eachline.split()[1:-1] 
            score_vals = [float(item) for item in score_str]

            textgrid_file = os.path.join(args.ali_dir, uttid + '.TextGrid')

            add_scores(score_vals, textgrid_file)

        print("The scores added to the TextGrid files!")
