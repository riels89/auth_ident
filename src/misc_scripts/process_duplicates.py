#!/usr/bin/python3

import sys

if len(sys.argv) != 2:
    print("Usage: ./process_duplicates <file_name>")

f = open(sys.argv[1])
s = f.read()
dups = s.split("\n\n")
count = 0
max = []
found = []
for dup in dups:
    lines = dup.split("\n")
    first_line = lines[0].split("/")
    if len(first_line) > 2:
        name = first_line[1]
        for line in lines:
            split_line = line.split("/")
            if name != split_line[1]:
                #print(dup, "\n")
                found.append(dup)
                count += len(lines)
                if len(lines)> len(max):
                    max = lines
                break
    else:
        print("Skipped")

#print("\n".join(max))
found.sort(key=len)
print("\n\n".join(found))
print("Total dups: ", count)
