# diversification-code
------------------------------------------------------------------------------------------
	                   Readme for the GSEMO algorithm Package	          
------------------------------------------------------------------------------------------

A. General information:

This package includes the python code of the Pareto Optimization method for Result Diversification. We divided them into six folders according to the experiment, named Experiment_synthetic, Experiment_letor, Experiment_enron_medical, Experiment_dynamic, Experiment_DUC2004 and Experiment_comments. Each file contains the necessary algorithms and data.

*************************************************************

greedy_ls.py: We run greedy and Local Search algorithm with it.

main.py: We run GSEMO algorithm with it.

GSEMO.py: The implementation of GSEMO algorithm.

In 'GSEMO.py', we declare a class GSEMO, which mainly contains the following functions:

setIterationTime(self, time): Set the iteration time.

mutation(self, s): Mutation operator flipping each bit of x with probability 1/n.

Calucalate_true_value(self, res): Calucalute the objective function value.

evaluateObjective(self, offspring): Evaluate the first objective value.

doGSEMO(self, path): Run the GSEMO algorithm.

*************************************************************

All of the experiments are coded in Python 3.8 and run on an identical configuration: a server with 2 Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60Ghz CPU and 128GB of RAM. The kernel is Linux 5.4.0-62-generic x86_64.

------------------------------------------------------------------------------------------