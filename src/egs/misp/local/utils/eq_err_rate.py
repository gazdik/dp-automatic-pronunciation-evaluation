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

import numpy as np
import argparse
import math
import err_rate as er
import helpers as hlp

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('flags', help='mispronunciation flags')
    parser.add_argument('scores', help='mispronunciation score in binary form')
    parser.add_argument('--trend', type=er.MispTrend, choices=list(er.MispTrend),
                        help='whether is mispronunciation score increasing or decreasing')
    parser.add_argument('--per_phone', nargs=2, metavar=('text', 'phn_map'),
                        default=(None, None))
    parser.add_argument('--thr_start', type=float, default=1.0, help='threshold starting value (default 1.0)')
    parser.add_argument('--thr_stop', type=float, default=0.0, help='the end value of threshold (default 0.0)')
    parser.add_argument('--thr_num', type=int, default=2000, help='number of threshold samples (default 200)')
    parser.add_argument('--thr', type=float)

    args = parser.parse_args()

    if args.thr:
        thresholds = np.array([args.thr])
    else:
        thresholds = np.linspace(args.thr_start, args.thr_stop, args.thr_num)

    text_fname, phn_map_fname = args.per_phone

    if text_fname:
        inv_phn_map = hlp.read_inv_phone_map(phn_map_fname)
        far, frr = er.err_per_phone(args.flags, args.scores, thresholds,
                                    text_fname, inv_phn_map, args.trend)
        # er.print_err_per_phone(thresholds, far, frr, inv_phn_map)
        eq_err_rates, eq_thresholds, atols = er.equal_err_per_phone(far, frr, thresholds)
        for i in range(len(eq_err_rates)):
            if math.isclose(eq_err_rates[i], 0.0, abs_tol=1e-5):
                continue
            print("%-10s" % (inv_phn_map[i]), "EER: %7.5f %%," % (eq_err_rates[i] * 100), "THR: %5.3f" % eq_thresholds[i],
                  "atol: %6.5f" % atols[i])
    else:
        far, frr = er.err(args.flags, args.scores, thresholds, args.trend)
        eq_err_rate, eq_threshold, atol = er.equal_err(far, frr, thresholds)
        print("All phones: ", "EER: %7.5f %%," % (eq_err_rate * 100),
              "THR: %5.3f" % eq_threshold,
              "atol: %6.5f" % atol)

