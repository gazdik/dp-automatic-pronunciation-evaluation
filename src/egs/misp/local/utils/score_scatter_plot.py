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

import kaldi_io as kio
import numpy as np
import enum
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matplotlib2tikz import save as tikz_save
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, expit

F_CORRECT = 0
F_SUBST = 1
F_DELET = 2


class Transformation(enum.Enum):
    none = 'none'
    inv_sig = 'inv_sig'
    log = 'log'
    exp = 'exp'

    def __str__(self):
        return self.value


trans_fn = {
    'none': np.copy,
    'inv_sig': logit,
    'log': np.log,
    'exp': np.exp,
}

parser = ArgumentParser()
parser.add_argument('score1')
parser.add_argument('score2')
parser.add_argument('text')
parser.add_argument('flags')
parser.add_argument('--p1', type=int)
parser.add_argument('--p2', type=int)
parser.add_argument('--trans_1', dest='trans_1', type=Transformation,
                    choices=list(Transformation), default=Transformation.none)
parser.add_argument('--trans_2', dest='trans_2', type=Transformation,
                    choices=list(Transformation), default=Transformation.none)
args = parser.parse_args()

score1_dict = {k:v for k,v in kio.read_vec_flt_ark(args.score1)}
score2_dict = {k:v for k,v in kio.read_vec_flt_ark(args.score2)}

trans1_fn = trans_fn[args.trans_1.value]
trans2_fn = trans_fn[args.trans_2.value]

text_dic = {k:v for k,v in kio.read_vec_int_ark(args.text)}

score_subst = []
score_delet = []
score_correct = []

for key, flags in kio.read_vec_int_ark(args.flags):
    text = text_dic[key]
    score1 = trans1_fn(score1_dict[key])
    score2 = trans2_fn(score2_dict[key])
    score1[score1 == 0.0] = 1e-10
    score2[score2 == 1.0] = 0.9999999999

    for s1, s2, f, p in zip(score1, score2, flags, text):
        if args.p1 and (p != args.p1 and p != args.p2):
            continue
        # if args.p2 and p != args.p2:
        #     continue
        if f == F_CORRECT:
            score_correct.append((s1, s2))
        if f == F_SUBST:
            score_subst.append((s1, s2))
        if f == F_DELET:
            score_delet.append((s1, s2))

# score_total = np.array(score_correct + score_subst + score_delet)
# norm = MinMaxScaler().fit(score_total)

score_subst = np.array(score_subst)
score_delet = np.array(score_delet)
score_correct = np.array(score_correct)
# score_subst = norm.transform(np.array(score_subst))
# score_delet = norm.transform(np.array(score_delet))
# score_correct = norm.transform(np.array(score_correct))

# np.random.shuffle(score_subst)
# np.random.shuffle(score_delet)
# np.random.shuffle(score_correct)

# z = np.polyfit(score_delet[:,0], score_delet[:,1], 1)
# p = np.poly1d(z)

plt.scatter(score_correct[:, 0], score_correct[:, 1], color='g', s=5)
plt.scatter(score_subst[:, 0], score_subst[:, 1], color='r', s=5)
plt.scatter(score_delet[:, 0], score_delet[:, 1], color='b', s=5)

# plt.scatter(score_correct[:500, 0], score_correct[:500, 1], color='g', s=5)
# plt.scatter(score_subst[:500, 0], score_subst[:500, 1], color='r', s=5)
# plt.scatter(score_delet[:500, 0], score_delet[:500, 1], color='b', s=5)



# tikz_save('score_scatter_plot.tex')
plt.show()
