#first import the packages needed
import numpy as np
import os
import argparse

#create a function
def file_loader():
    #initialize a parser object
    parser = argparse.ArgumentParser(description="Loading and printing an array")
    parser.add_argument("--input",
                        "-i",
                        required = True,
                        help ="Filepath to CSV to load and print")

    args = parser.parse_args()
    return args

def main():
    args = file_loader()
    #create filepath to data
    filename = os.path.join("..","..","..","..","cds-vis-data","data","sample-data",args.input)
    
    #load data
    data = np.loadtxt(filename, delimiter = ",")


    print(data)

# unless its run in an command line, it will do nothing
if __name__=="__main__":
    main()