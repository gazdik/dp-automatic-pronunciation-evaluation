#!/usr/bin/python3

import sys

if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) < 2:
    raise TypeError ('USAGE: create_symbol_table.py <input-symbols> [<old-table>]')

symbols_file   = sys.argv[1]
old_table_file = None
if len(sys.argv) == 3:
    old_table_file = sys.argv[2]

counter = 0
dict = {}

# Read the old table if available
if old_table_file is not None:
    with open(old_table_file, mode='r') as f:
        for line in f:
            key, val = line.split()
            dict[key] = int(val)

# Update the counter
counter = len(dict)

# Read input symbols
symbols = []
with open(symbols_file) as f:
    for line in f:
        symbols.append(line.split()[0])

# Add new symbols into the dictionary
for s in symbols:
    if s not in dict:
        dict[s] = counter
        counter = counter + 1
        
# Write the dictionary to the output
for key, val in dict.items():
    print("%s %d" % (key, val))