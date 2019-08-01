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
from pfeats_map import PFeatsMap
import generator_interface as gi
from subprocess import Popen, PIPE


class DataLoader(gi.MultiOutDataLoader):
    def __init__(self, input_stream, pdf_stream, phone_stream, exp_dir,
                 phone_map_fname, pfeats_lang, tmp_dir, batch_size):
        self.pfeats_map = PFeatsMap(phone_map_fname, pfeats_lang)
        self.exp_dir = exp_dir

        input_it = kio.read_mat_ark(input_stream)
        pdf_dict = {k:v for k,v in kio.read_ali_ark(pdf_stream)}
        phone_dict = {k:v for k,v in kio.read_ali_ark(phone_stream)}

        print("Loading data ", tmp_dir)
        gi.MultiOutDataLoader.__init__(self, input_it, (pdf_dict, phone_dict),
                                       tmp_dir, batch_size)
        print("Data loaded", tmp_dir)

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            try:
                key, self.input_frames = next(self.input_it)
            except StopIteration:
                return None, None

            self.idx = 0
            for i in range(self.num_outputs):
                self.output_frames[i] = self.output_dict[i][key]

        inputs = self.input_frames[self.idx]
        outputs = [self.output_frames[0][self.idx],
                   self.pfeats_map.phn_to_pfeats(self.output_frames[1][self.idx])]
        self.idx += 1

        return inputs, outputs

    def output_dim(self):
        """ Returns output dimension """
        return [self.pdf_output_dim(), self.pfeats_map.pfeats_dim()]

    def pdf_output_dim(self):
        """ Returns output dimension """
        p = Popen(['am-info', '%s/final.mdl' % self.exp_dir], stdout=PIPE)
        model_info = p.stdout.read().splitlines()
        for line in model_info:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

        raise EOFError("pdf_generator.py: Couldn't parse model info file")


class DataGenerator(gi.DataGenerator):
    def __init__(self, input_stream, pdf_stream, phone_stream, exp_dir,
                 phone_map_fname, pfeats_lang, tmp_dir, batch_size=256,
                 shuffle=True):
        data_loader = DataLoader(input_stream, pdf_stream, phone_stream,
                                 exp_dir, phone_map_fname,
                                 pfeats_lang, tmp_dir, batch_size)

        gi.DataGenerator.__init__(self, data_loader, shuffle)
