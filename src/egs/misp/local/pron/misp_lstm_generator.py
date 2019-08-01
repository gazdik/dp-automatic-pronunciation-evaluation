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
import random
from helpers import read_phone_map, get_segment_idxs
from keras.preprocessing.sequence import pad_sequences
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

    def load_next_utterance(self):
        try:
            while True:  # TODO workaround solving problem with missing alignments
                key, input_frames = next(self.input_it)
                if key not in self.output_dict:
                    continue
                break
        except StopIteration:
            return None

        self.idx = 0
        output_frames = self.output_dict[key]

        # TODO Group segments using different alignments... because
        # TODO This can lead to merging two identical subsequent phonemes
        # Group frames into segments
        segment_idxs = get_segment_idxs(output_frames)
        self.input_frames = np.split(input_frames, segment_idxs)
        self.output_frames = np.split(output_frames, segment_idxs)
        self.flags = self.flags_dict[key]

        return key

    def init_data(self):
        # Load the first frame
        self.idx = 0
        key, input_frames = next(self.input_it)
        output_frames = self.output_dict[key]

        # Group frames into segments
        segment_idxs = get_segment_idxs(output_frames)
        self.input_frames = np.split(input_frames, segment_idxs)
        self.output_frames = np.split(output_frames, segment_idxs)

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
            output_list.append(self.to_label(output_frame[0], flag))

        if len(input_list) == 0:
            return None

        return pad_sequences(input_list), np.vstack(output_list)

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            key = self.load_next_utterance()
            if key is None:
                return None, None, None

        input = self.input_frames[self.idx]
        output = self.output_frames[self.idx]
        flag = self.flags[self.idx]
        self.idx += 1

        return input, output, flag

    def output_dim(self):
        return len(self.phone_map)

    def input_dim(self):
        return len(self.input_frames[0][0])

    def to_label(self, phone: int, flag: int) -> np.array:
        label = np.full(self.output_dim(), -1, dtype=np.int)

        if flag == 0:
            label[self.phone_map[phone]] = 1
        else:
            label[self.phone_map[phone]] = 0

        return label

    def scatter(self):
        """ Scatter elements between batches to increase randomness """
        idxs = self.batch_idxs.copy()
        random.shuffle(idxs)

        for i in range(0, len(idxs) - 1, 2):
            batch1 = self.get_batch(idxs[i])
            batch2 = self.get_batch(idxs[i + 1])

            # Pad sequences since they may have different lengths
            batch1 = list(batch1); batch2 = list(batch2)  # Workaround to assign value to tuple
            maxlen = max(batch1[0].shape[1], batch2[0].shape[1])
            batch1[0] = pad_sequences(batch1[0], maxlen=maxlen)
            batch2[0] = pad_sequences(batch2[0], maxlen=maxlen)
            batch1 = tuple(batch1); batch2 = tuple(batch2)

            gi.scatter_batches(batch1, batch2)

            self.save_batch(idxs[i], batch1)
            self.save_batch(idxs[i + 1], batch2)

class DataGenerator(gi.DataGenerator):
    def __init__(self, input_fname, output_fname, flags_fname, phone_map_fname,
                 tmp_dir, batch_size=256, shuffle=True):
        data_loader = DataLoader(input_fname, output_fname, flags_fname, phone_map_fname,
                                 tmp_dir, batch_size)

        gi.DataGenerator.__init__(self, data_loader, shuffle)

