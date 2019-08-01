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
import numpy_indexed as npi
from itertools import islice
import sys, os, re, gzip, struct


def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'r') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'w') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd


# based on '/usr/local/lib/python3.6/os.py'
def popen(cmd, mode="rb"):
    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    import subprocess, io, threading

    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise Exception('cmd %s returned %d !' % (cmd,ret))
        return

    # text-mode,
    if mode == "r":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    elif mode == "w":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    elif mode == "rb":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdout
    elif mode == "wb":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdin
    # sanity,
    else:
        raise ValueError("invalid mode %s" % mode)


def read_segment_lengths(alignment_fname):
    fd = open_or_fd(alignment_fname, mode='r')
    try:
        line = fd.readline()
        while line:
            tokens = line.strip().split(" ")
            key = tokens.pop(0)
            seg_lengths = list(islice(tokens, 1, None, 3))
            seg_lengths = np.array(seg_lengths, dtype=np.int)
            yield key, seg_lengths
            line = fd.readline()
    finally:
        fd.close()


def read_phone_alignments(alignments_fname):
    fd = open_or_fd(alignments_fname, mode='r')
    try:
        line = fd.readline()
        while line:
            tokens = line.strip().split(" ")
            key = tokens.pop(0)
            phone_ali = list(islice(tokens, 0, None, 3))
            phone_ali = np.array(phone_ali, dtype=np.int)
            yield key, phone_ali
            line = fd.readline()
    finally:
        fd.close()


def read_text_file(text_fname):
    fd = open(text_fname, mode='r')
    try:
        line = fd.readline()
        while line:
            tokens = line.strip().split()
            key = tokens.pop(0)
            yield key, tokens
            line = fd.readline()
    finally:
        fd.close()


def utterance_equal(utt1, utt2):
    if len(utt1) != len(utt2):
        return False

    for x, y in zip(utt1, utt2):
        if x != y:
            return False

    return True


def utterance2str(utterance):
    return " ".join(map(str,utterance))


def get_segment_idxs(phone_labels):
    return np.where(np.roll(phone_labels, 1) != phone_labels)[0]


def read_phone_map(phone_map_fname):
    phone_map = {}
    with open(phone_map_fname, mode='r') as f:
        for line in f:
            key, val = line.split()
            phone_map[int(key)] = int(val)

    return phone_map


def read_inv_phone_map(phone_map_fname):
    phone_map = {}
    with open(phone_map_fname, mode='r') as f:
        for line in f:
            key, val = line.split()
            phone_map[int(val)] = key

    return phone_map


def get_seg_lengths(ali: np.array) -> np.array:
    """
    Get length of each segment.
    Return: Numpy array of lengths per segment.
    """
    i = 0
    lengths = []
    prev_phone = ali[0]

    for phone in ali:
        if phone != prev_phone:
            prev_phone = phone
            lengths.append(i)
            i = 0
        i = i + 1
    lengths.append(i)

    return np.array(lengths)


def get_seg_idx(seg_lengths: np.array) -> np.array:
    """
    Calculate segment indexes based on segment lengths
    """
    seg_idx = np.cumsum(seg_lengths)
    seg_idx = np.insert(seg_idx, 0, 0)
    return seg_idx


def flt_vec_to_str(key, flt_arr):
    str = key
    for i in np.nditer(flt_arr):
        str = str + (" %.2f" % i)
    
    return str


def int_vec_to_str(key, arr):
    str = key
    for i in np.nditer(arr):
        str = str + (" %5d" % i)
    
    return str


def eq_flt_vec_to_str(key, flt_arr):
    str = key
    for i in np.nditer(flt_arr):
        str = str + (" %5.2f" % i)
    
    return str


def np_pick(arr, indexes):
    return arr[np.arange(len(arr)), indexes]


class Pdf2PhnFeats:
    def __init__(self, phn_to_pdf_fname):
        pdf_to_phn = read_inv_phone_map(phn_to_pdf_fname)
        pdf_to_phn_arr = [int(pdf_to_phn[x]) for x in range(len(pdf_to_phn))]
        self.phn_dim = np.max(pdf_to_phn_arr) + 1
        self.group_by_phn = npi.group_by(pdf_to_phn_arr)

    def convert(self, pdf_array):
        pdf_array = np.exp(pdf_array)
        keys, vals = self.group_by_phn.sum(pdf_array, axis=1)
        vals = np.log(vals)
        phn_arr = np.full((len(vals), self.phn_dim), float('nan'))

        for i, key in enumerate(keys):
            phn_arr[:, key] = vals[:, i]

        return phn_arr


