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
import err_rate as er
import helpers as hlp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save


def roc_curve(far, frr, out_file=None):
    """Plot DET curve from given FAR and FRR rates
    The FAR rates are expected to be increasing while the FRR rates
    are expected to be decreasing.
    """
    far_p = far * 100
    frr_p = frr * 100

    fig, ax = plt.subplots()
    ax.get_xaxis().set_major_formatter(mpl.ticker.PercentFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.PercentFormatter())
    plt.xlabel('FRR')
    plt.ylabel('FAR')

    line = np.linspace(0, 100, num=len(far))
    plt.plot(line, line)
    plt.plot(frr_p, far_p)

    if out_file:
        print("Saving plot %s" % out_file)
        tikz_save(out_file + ".tex")
        plt.savefig(out_file + ".png")

        plt.close(fig)
    else:
        plt.show()

def roc_curve_per_phone(far, frr, inv_phn_map, out_file=None):
    far_p = far * 100
    frr_p = frr * 100

    fig, ax = plt.subplots()
    ax.get_xaxis().set_major_formatter(mpl.ticker.PercentFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.PercentFormatter())
    plt.xlabel('FRR')
    plt.ylabel('FAR')

    line = np.linspace(0, 100, num=len(far))
    plt.plot(line, line)

    legends = []
    for i in range(len(far)):
        if np.any(np.isnan(far[i])):
            continue

        legend, = plt.plot(frr_p[i], far_p[i], label=inv_phn_map[i])
        legends.append(legend)

    plt.legend(ncol=2)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)
    # plt.legend(legends)

    if out_file:
        print("Saving plot %s" % out_file)
        tikz_save(out_file + ".tex")
        plt.savefig(out_file + ".png")

        plt.close(fig)
    else:
        plt.show()


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('flags', help='mispronunciation flags')
parser.add_argument('scores', help='mispronunciation score in binary form')
parser.add_argument('misp_trend', type=er.MispTrend, choices=list(er.MispTrend),
                    help='whether is mispronunciation score increasing or decreasing')
parser.add_argument('--thr_num', type=int, default=200, help='number of threshold samples (default 200)')
parser.add_argument('--thr_start', type=float, default=1.0, help='threshold starting value (default 1.0)')
parser.add_argument('--thr_stop', type=float, default=0.0, help='the end value of threshold (default 0.0)')
parser.add_argument('--out_file', help='output file')
parser.add_argument('--per_phone', nargs=2, metavar=('text', 'phn_map'),
                    default=(None, None), help='ros curves per phone')
  
args = parser.parse_args()

text_fname, phn_map_fname = args.per_phone

# Calculate FAR and FRR rates
thresholds = np.linspace(args.thr_start, args.thr_stop, args.thr_num)
if text_fname:
    phn_map = hlp.read_inv_phone_map(phn_map_fname)
    far, frr = er.err_per_phone(args.flags, args.scores, thresholds,
                                text_fname, phn_map, args.misp_trend)
    roc_curve_per_phone(far, frr, phn_map, out_file=args.out_file)

else:
    far, frr = er.err(args.flags, args.scores, thresholds, args.misp_trend)
    roc_curve(far, frr, out_file=args.out_file)







