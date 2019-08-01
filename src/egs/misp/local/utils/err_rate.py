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

import kaldi_io as kio
import numpy as np
import argparse
import helpers as hlp
import enum
import math


class MispTrend(enum.Enum):
    inc = 'inc'
    dec = 'dec'

    def __str__(self):
        return self.value


misp_fn = {MispTrend.inc: (lambda score, thr: score > thr), MispTrend.dec: (lambda score, thr: score < thr)}

def err(flags_fname, scores_fname, threshold, misp_trend):
    flags_dict = { k:v for k,v in kio.read_vec_int_ark(flags_fname) }

    far = np.zeros_like(threshold, dtype=np.int32)
    frr = np.zeros_like(threshold, dtype=np.int32)
    cnt_mp = 0
    cnt_cp = 0
    for uttid, scores in kio.read_vec_flt_ark(scores_fname):
        flags = flags_dict[uttid]

        for i in range(len(flags)):
            if flags[i] > 2:
                continue

            mp = misp_fn[misp_trend](scores[i], threshold)

            if flags[i] == 0:
                cnt_cp += 1
                # correct[mp == False] = correct[mp == False] + 1
                frr[mp == True] += 1
            elif flags[i] > 0:
                cnt_mp = cnt_mp + 1
                # correct[mp == True] = correct[mp == True] + 1
                far[mp == False] = far[mp == False] + 1
            else:
                print("Imposible?!")

    return (far / cnt_mp), (frr / cnt_cp)


def err_per_phone(flags_fname, scores_fname, threshold, text_fname, phn_map, misp_trend):
    flags_dict = { k:v for k,v in kio.read_vec_int_ark(flags_fname) }
    text_dict = { k:v for k,v in kio.read_vec_int_ark(text_fname) }

    far = np.zeros((len(phn_map), len(threshold)), dtype=np.int32)
    frr = np.zeros((len(phn_map), len(threshold)), dtype=np.int32)
    cnt_mp = np.zeros(len(phn_map), dtype=np.int32)
    cnt_cp = np.zeros(len(phn_map), dtype=np.int32)

    for utt_id, scores in kio.read_vec_flt_ark(scores_fname):
        flags = flags_dict[utt_id]
        phones = text_dict[utt_id]

        for i in range(len(flags)):
            if flags[i] > 2:
                continue

            phn = phones[i]
            mp = misp_fn[misp_trend](scores[i], threshold)

            if flags[i] == 0:
                cnt_cp[phn] += 1
                frr[phn, mp == True] = frr[phn, mp == True] + 1
            elif flags[i] > 0:
                cnt_mp[phn] += 1
                far[phn, mp == False] = far[phn, mp == False] + 1

    cnt_mp = cnt_mp.astype(np.float)
    cnt_cp = cnt_cp.astype(np.float)
    cnt_mp[cnt_mp == 0.] = float('nan')
    cnt_cp[cnt_cp == 0.] = float('nan')
    cnt_mp = np.reshape(cnt_mp, (-1, 1))
    cnt_cp = np.reshape(cnt_cp, (-1, 1))
    return (far / cnt_mp), (frr / cnt_cp)


def equal_err(far, frr, thresholds):
    eq_err_rate = float('nan')
    eq_threshold = float('nan')
    abs_tol = float('nan')

    for atol in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        foo = np.isclose(far, frr, atol=atol)
        if not foo.any():
            continue
        eq_err_rate = far[foo].item(0)
        eq_threshold = thresholds[foo].item(0)
        abs_tol = atol
        break

    return eq_err_rate, eq_threshold, abs_tol


def equal_err_per_phone(far, frr, thresholds):
    eq_err_rate = np.full(len(far), np.nan)
    eq_thresholds = np.full(len(far), np.nan)
    atols = np.full(len(far), np.nan)

    for i in range(len(far)):
        for atol in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            foo = np.isclose(far[i], frr[i], atol=atol)
            if not foo.any():
                continue
            eq_err_rate[i] = far[i, foo].item(0)
            eq_thresholds[i] = thresholds[foo].item(0)
            atols[i] = atol
            break

    return eq_err_rate, eq_thresholds, atols


def print_err_per_phone(thresholds, far, frr, inv_phn_map):
    for phone in range(len(far)):
        if np.any(np.isnan(far[phone])):
            continue

        print("Phone: %-8s" % inv_phn_map[phone])
        print_err(thresholds, far[phone], frr[phone])


def print_err(thresholds, far, frr):
    print("  Thr:    ", end=' ')
    for i in np.nditer(thresholds):
        print("%.3f" % i, end=' ')
    print("\n  FAR:    ", end=' ')
    for i in np.nditer(far):
        print("%.3f" % i, end=' ')
    print("\n  FRR:    ", end=' ')
    for i in np.nditer(frr):
        print("%.3f" % i, end=' ')
    print("")


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('flags', help='mispronunciation flags')
    parser.add_argument('scores', help='mispronunciation score in binary form')
    parser.add_argument('misp_trend', type=MispTrend, choices=list(MispTrend),
                        help='whether is mispronunciation score increasing or decreasing')
    parser.add_argument('thresholds', nargs='+', type=float, help='one or more thresholds')
    parser.add_argument('--per_phone', nargs=2, metavar=('text', 'phn_map'),
                        default=(None, None))
    args = parser.parse_args()

    thresholds = np.array(args.thresholds, dtype=np.float)
    text_fname, phn_map_fname = args.per_phone

    if text_fname:
        inv_phn_map = hlp.read_inv_phone_map(phn_map_fname)
        far, frr = err_per_phone(args.flags, args.scores, thresholds,
                                 text_fname, inv_phn_map, args.misp_trend)
        print_err_per_phone(thresholds, far, frr, inv_phn_map)
    else:
        far, frr = err(args.flags, args.scores, thresholds, args.misp_trend)
        print_err(thresholds, far, frr)

