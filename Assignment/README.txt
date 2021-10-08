AUTHOR: Nicholas Klvana-Hooper
STUDENT ID: 19872944
Date Created: 8/10/2021

This is a readme for the Assignment for Data Mining COMP3009 taken during the second semester of 2021.

The main file for this assignment is the run.py file. This contains all of the necessary code to run the program. This is paired with the
data2021.student.csv file which contains the data for the run.py file to work. Currently a predict.csv file exists which will will be overwritten with
new values.

The following packages were used:
pandas, numpy, sklearn, matplotlib, imblearn

How the code works:
The main function should get called straight away.
It runs functions that deal with preProcessing straight away, including looking for duplicates, missing data etc.
After this it splits the data into sets to test and train on
There is one commented out function here called modelTuning which looks at tuning hyperparameters for models for classifcation. Since this isn't needed
in the final submission and takes time I have commented it out in the main function.
From there theere is a comparison between models, and then the final prediction comparison which handles the creating of the output file.

Problems with the code:
There is warning that will be generated everytime the program runs warning about overwriting pandas with slices. This can be ignored and doesn't affect
this program in any way. It should not stop the program fromn running.