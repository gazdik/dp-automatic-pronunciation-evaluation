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
from pfeats_map import PFeatsMap
import generator_interface as gi


class DataLoader(gi.DataLoader):
    def __init__(self, input_stream, output_stream, exp_dir, phone_map_fname, pfeats_lang, tmp_dir, batch_size):
        self.pfeats_map = PFeatsMap(phone_map_fname, pfeats_lang)

        input_it = kio.read_mat_ark(input_stream)
        output_dict = {k:v for k,v in kio.read_ali_ark(output_stream)}

        gi.DataLoader.__init__(self, input_it, output_dict, tmp_dir, batch_size)

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            try:
                key, self.input_frames = next(self.input_it)
            except StopIteration:
                return None, None

            self.idx = 0
            self.output_frames = self.output_dict[key]

        inputs = self.input_frames[self.idx]
        outputs = self.pfeats_map.phn_to_pfeats(self.output_frames[self.idx])
        self.idx += 1

        return inputs, outputs

    def output_dim(self):
        """ Returns output dimension """
        return self.pfeats_map.pfeats_dim()


class DataGenerator(gi.DataGenerator):
    def __init__(self, input_dir, output_dir, exp_dir, phone_map_fname, pfeats_lang, tmp_dir, batch_size=256,
                 shuffle=True):
        data_loader = DataLoader(input_dir, output_dir, exp_dir, phone_map_fname,
                                 pfeats_lang, tmp_dir, batch_size)

        gi.DataGenerator.__init__(self, data_loader, shuffle)

