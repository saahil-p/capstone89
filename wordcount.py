#! /usr/bin/env python3

import sys

ip = sys.stdin.read()

words = ip.strip().split()

print(len(words))