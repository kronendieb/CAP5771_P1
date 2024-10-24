# Project 1 for CAP5771 Principles of Data Mining.

### Important information
    This project requires python 3.12.
    Use 'pip install -r requirements.txt' to install all dependencies.
    To run the project use: python team01.py "<filename>" <minsuppc> <minconf>
        where:
            filename is the name of the input file formated in rows of "index value" pairs
            minsuppc is the minimum accepted support count, must be above 0, and lower numbers may not be runnable on large data
            minconf  is the minimum confidence accepted to generate rules, must be equal or above 0. a value of -1 does not generate rules.
    To run the example small.txt try: python team01.py "small.txt" 200 0.5

### Files in the project
    README.md        -- This file.
    team01.py          -- This is the file where all of the python code lies.
    requirements.txt -- The python packages required to run the code.
    small.txt        -- Example file for the project testing.
    items01.txt      -- Output file for the frequent items found
    rules01.txt      -- Output file for the high confidence rules found
    info01.txt       -- Output file for the info gathered about the program.
    shell.nix        -- Operating system file, do not touch, do not submit.
    
