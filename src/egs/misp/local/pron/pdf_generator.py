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
import generator_interface as gi

from subprocess import Popen, PIPE


class DataLoader(gi.DataLoader):
    def __init__(self, input_stream, output_stream, exp_dir, tmp_dir, batch_size):
        self.exp_dir = exp_dir

        input_it = kio.read_mat_ark(input_stream)
        output_dict = {k:v for k,v in kio.read_ali_ark(output_stream)}

        gi.DataLoader.__init__(self, input_it, output_dict, tmp_dir, batch_size)

    def output_dim(self):
        """ Returns output dimension """
        p = Popen(['am-info', '%s/final.mdl' % self.exp_dir], stdout=PIPE)
        model_info = p.stdout.read().splitlines()
        for line in model_info:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

        raise EOFError("pdf_generator.py: Couldn't parse model info file")


class DataGenerator(gi.DataGenerator):
    def __init__(self, input_dir, output_dir, exp_dir, tmp_dir, batch_size=256,
                 shuffle=True):
        data_loader = DataLoader(input_dir, output_dir, exp_dir, tmp_dir, batch_size)

        gi.DataGenerator.__init__(self, data_loader, shuffle)


