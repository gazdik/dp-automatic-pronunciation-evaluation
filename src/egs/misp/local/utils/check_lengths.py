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

import sys
import kaldi_io as kio
from argparse import ArgumentParser
import enum


class DataType(enum.Enum):
    int_ark = 'int_ark'
    flt_ark = 'flt_ark'
    mat_ark = 'mat_ark'

    def __str__(self):
        return self.value


kio_fn = {
    'int_ark': kio.read_vec_int_ark, 
    'flt_ark': kio.read_vec_flt_ark, 
    'mat_ark': kio.read_mat_ark
}

# Parse arguments
parser = ArgumentParser()
parser.add_argument('file_1', help='the first ark file')
parser.add_argument('file_2', help='the second ark file')
parser.add_argument('--type_1', dest='type_1', type=DataType, choices=list(DataType), default=DataType.int_ark)
parser.add_argument('--type_2', dest='type_2', type=DataType, choices=list(DataType), default=DataType.int_ark)
args = parser.parse_args()

# Open both files
iter1 = kio_fn[args.type_1.value](args.file_1)
iter2 = kio_fn[args.type_2.value](args.file_2)

equal = True
print("Comparing the lengths of", args.file_1, "and", args.file_2)
for x, y in zip(iter1, iter2):
    if x[0] != y[0]:
        raise Exception("Files contains different keys or aren't sorted.")

    if len(x[1]) != len(y[1]):
        print("{}: {} != {}".format(x[0], len(x[1]), len(y[1])))

if not equal:
    sys.exit(1)
