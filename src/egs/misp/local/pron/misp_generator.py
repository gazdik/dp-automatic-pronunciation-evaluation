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

import kaldi_io as kio
import numpy as np
from helpers import read_phone_map
import generator_interface as gi


class DataLoader(gi.DataLoader):
    def __init__(self, input_stream, output_stream, flags_stream, phone_map_fname,
                 tmp_dir, batch_size):
        self.phone_map = read_phone_map(phone_map_fname)

        input_it = kio.read_mat_ark(input_stream)
        output_dict = {k: v for k, v in kio.read_ali_ark(output_stream)}
        self.flags_dict = {k: v for k, v in kio.read_vec_int_ark(flags_stream)}

        self.flags = None

        gi.DataLoader.__init__(self, input_it, output_dict, tmp_dir, batch_size)

    def init_data(self):
        # Load the first frame
        self.idx = 0
        key, self.input_frames = next(self.input_it)
        self.output_frames = self.output_dict[key]
        self.flags = self.flags_dict[key]

    def get_next_batch(self):
        input_list = []
        output_list = []

        while len(input_list) < self.batch_size:
            input_frame, output_frame, flag = self.get_next_frame()

            if input_frame is None:
                break
            elif flag > 2:
                continue

            input_list.append(input_frame)
            output_list.append(self.to_label(output_frame, flag))

        if len(input_list) == 0:
            return None

        return np.vstack(input_list), np.vstack(output_list)

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            try:
                key, self.input_frames = next(self.input_it)
            except StopIteration:
                return None, None, None

            self.idx = 0
            self.output_frames = self.output_dict[key]
            self.flags = self.flags_dict[key]

        input = self.input_frames[self.idx]
        output = self.output_frames[self.idx]
        flag = self.flags[self.idx]
        self.idx += 1

        return input, output, flag

    def output_dim(self):
        """ Returns output dimension """
        return len(self.phone_map)

    def to_label(self, phone: int, flag: int) -> np.array:
        label = np.full(self.output_dim(), -1, dtype=np.int)

        if flag == 0:
            label[self.phone_map[phone]] = 1
        else:
            label[self.phone_map[phone]] = 0

        return label


class DataGenerator(gi.DataGenerator):
    def __init__(self, input_fname, output_fname, flags_fname, phone_map_fname,
                 tmp_dir, batch_size=256, shuffle=True):
        data_loader = DataLoader(input_fname, output_fname, flags_fname, phone_map_fname,
                                 tmp_dir, batch_size)

        gi.DataGenerator.__init__(self, data_loader, shuffle)

