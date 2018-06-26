# -*- coding: utf-8 -*-
"""
author: GC
Run script if a raw domoticz safehouse JSON file isn't playing nicely with python's or panda's JSON functions.
May not be the best solution for large files.
TO DO:  Minimize memory footprint.
        Automate process for whole directory.
        Pasre ifttt data.
"""
import io

# Set the path to your data directory
path = '/home/helio/Group-4-Project/data/domo/'

# functions just add commas and brackets appropriately.
def process_line(l):    
    return l.replace("}}","}},")

def restore_line(l):
    return l.replace("}},","}}")

# open files to read and write
infile = open(path + "domoticz-2018-04-10.json", "r")
data = infile.readlines()
outfile = open(path + "domoticz-2018-04-10-cleaned.json", "w")
outfile.write("[\n")
for i in range(len(data[:-1])):
    data[i] = process_line(data[i])
    
outfile.writelines(data[:-1])
outfile.writelines(data[-1])
outfile.write("]")
    
infile.close()
outfile.close()
     