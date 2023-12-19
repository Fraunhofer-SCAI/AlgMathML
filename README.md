# AlgMathML

This repository contains the template notebooks and data sets for the tasks from the text book "Algorithmic Mathematics in Machine Learning" by Bastian Bohn, Jochen Garcke and Michael Griebel.

How to install jupyter-notebooks and get the necessary python packages (assuming
you are running a Linux-based operating system - in the case of different operating systems please check https://docs.python.org/3/library/venv.html to see how to set up a virtual environment appropriately):

1. Make sure you have python3 and pip3 installed.
2. Install virtualenv via 
	"sudo -H pip3 install virtualenv"
   or use python3's venv instead if you do not have virtualenv installed and do
   not have root rights.
3. Go to the directory of your choice and create a virtual environment via
	"virtualenv -p python3 .mlbook"
	or
	"python3 -m venv .mlbook"
4. Once created you should activate the environment by running
	"source .mlbook/bin/activate"
5. Then, to install the necessary packages for python download the
   requirements.txt file from the practical lab website and then run
	"pip3 install -r requirements.txt"
6. Finally, you can run and edit your jupyter-notebooks via
	"jupyter-notebook NameOfYourNotebook.ipynb"
7. When you want to leave the virutal environment use
	"deactivate"

If the above fails, try to use "requirements_w_version.txt" instead in step 5.


You can find examplary images depicting solutions to certain tasks in the folder [solution_example_pictures](solution_example_pictures). Note that your solution might look differently depending on your specific implementation and the specific training and test data set you used.

In case you are a lecturer and you are interested in reference solutions to these tasks, please contact us at bohn@ins.uni-bonn.de.
