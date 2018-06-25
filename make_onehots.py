"""
The CSV files from the Human Activity Recognition With Smartphones* dataset
contain string labels in their Activity column. To make things simpler for
Octave, I will convert these string labels into one-hots.

*: https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones 
"""
from operator import itemgetter

""" Returns a one-hot string, where the 1 is located on index 'one_location'.
The one-hot has 'length' number of items.
"""
def make_onehot(one_location, length):
    zeros = ["0"]*length
    zeros[one_location] = "1"
    return ",".join(zeros)

""" Returns a dictionary where each item in 'activities' is a dictionary key
that corresponds to a unique one-hot. Specifically, each item i in 'activities'
corresponds to a one-hot where the 1 is located at index i of the vector.
"""
def make_onehot_dict(activities):
    activity_dict = {}
    for i, activity in enumerate(activities):
        activity_dict[activity] = make_onehot(i, len(activities))

    return activity_dict

""" Replace the strings in 'filename_in''s activity column with one-hots. New
version of this CSV file with one-hots is written to 'filename_out'.

'filename_in' is a CSV file from the dataset:
https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones 
"""
def make_onehot_file(filename_in, filename_out, verbose=True):
    # Possible labels in the 'Activity' column of 'filename_in'
    activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",\
            "SITTING", "STANDING", "LAYING"]

    activity_dict = make_onehot_dict(activities)
    with open(filename_in, "r") as fin, open(filename_out, "w") as fout:
        # Giving output file a new header
        file_header = fin.readline().split(",")
        file_header[-1] = ",".join(activities) + "\n"
        fout.write(",".join(file_header))
   
        # Substituting string activities with one-hots
        for line in fin:
            items = line.split(",")
            activity = items[-1].strip().strip("\"")
            items[-1] = activity_dict[activity] + "\n"
            fout.write(",".join(items))

    if verbose:
        print ("String activities in {0} have been written as one-hots in"
                " {1}").format(filename_in, filename_out)

def main():
    make_onehot_file("./test.csv", "./test_onehot.csv")
    make_onehot_file("./train.csv", "./train_onehot.csv")

main()
