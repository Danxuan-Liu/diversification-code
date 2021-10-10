------------------------------------
Original dataset:

Now the Experiment_Date folder is empty, you can download the original dataset "MSLR-WEB10K" online .

------------------------------------
Operate dataset:

step 0: We integrated the downloaded data into the file "original_data.txt".

step 1: Run operate_data.py to generate the info_sort.txt which contains the information of qid whose number of documents are more than 370.

step 2: Run generate_top.py to get the instances with exactly 370 doucuments.

step 3: After step 2, you can get 82 instances, we only need 50 to do the experiment.