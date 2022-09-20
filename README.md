# Code for Paper *Model predicted human mobility explains COVID-19 transmission in urban space without behavioral data*

## System Requirement

#### Example System Information

Operating System: Ubuntu 18.04.5 LTS
CPU: AMD Ryzen Threadripper 2990WX 32-Core Processor
Memory: 128G DDR4 Memory

#### Tested Software Versions
Anaconda3-2021.05-Linux-x86_64

conda 4.7.12

python==3.7.0

important python packages:

numpy==1.21.6

matplotlib==3.4.3

datetime==4.3

pandas==1.1.5

bayesian-optimization==1.2.0

scikit-learn==0.24.2

scipy==1.6.3

seaborn==0.11.1

shapely==1.7.1

adjustText==0.7.3

setproctitle==1.3.2

### Installation Guide
Typically, a morden computer with fast internet can complete the installation within 10 mins.

1. Download Anaconda according to [Official Website](https://[www.anaconda.com/products/individual-d](https://www.anaconda.com/products/distribution)), which can be done by the fillowing command (newer version of anaconda is also OK)
``` bash
wget -c https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
```
2. Install Anaconda through the commandline guide. Permit conda init when asked.
``` bash
./Anaconda3-2021.05-Linux-x86_64.sh
```
3. Quit current terminal window and open a new one. You should able to see (base) before your command line. 

4. Use the following command to install pre-configured environment through the provided .yml file (you should go to the directory of this project before performing the command)
``` bash
conda env create -f ./anaconda_env_covid.yml
```

5. Finally, activate the installed environment. Now you can run the example code through the following chapter.
``` bash
conda activate covid
```

(Optional) If you need to exit the environment for other project, use the following command.

``` bash
conda deactivate 
```

(Optional) Command for creating our environment without the .yml file
``` bash
conda create -n covid python==3.7
pip install numpy ipython pandas matplotlib setproctitle bayesian-optimization datetime pathlib scikit-learn scipy adjustText seaborn shapely scikit-learn scipy seaborn
```

## Run the Codes

In this repo, we provide the fit process of our model (in ./fit) and the source code for analyses in the main text (in ./analyses).

### Reproducing the model fitting

The fitting process contains all the cities in our experiments, including **20** counties in U.S., **5** cities in India and **5** citeis in Brazil. Here, we provide the procedure to run these codes taking the Brazil city Sao Paulo as an example:

1. Activate the environment
``` bash
conda activate covid
```
2. Go to the corresponding directory
``` bash
cd ./fit/brazil
```
3. Run the following command
``` bash
python simulate_ode_fit_saoPaulo_BO.py
```

You will get the fitted results in simulated_results_SaoPaulo_2500 directory. It contains three kinds of results: a figure that shows the fitted & real cumulated infected curve (in green and red), a json file that records the corresponding fitted model parameters, and a npy file of the fitted cumulated infected curve.

simulate_ode_fit\_($cityname)\_BO.py are responsible for run the fitting for each city. You can try other cities as you wish.

Note that in our experiment, we fit 40 times to test our results, and we run the code by multi-processing with 20 runs. You may change the setting for higher or lower to suit your computation power. In our case, it often takes us a few hours for each city in the default setup.

In simulate_ode_fit\_($cityname)\_BO.py, you can find:
``` python
MULTI_PROCESSING = 20
```
``` python
if __name__ == "__main__":
    fit_num = 40
```


### Reproducing Figures in Main Text

We also include all the experiments in our main text in the ./analyses folder. Since the simulation results are relatively huge, we provide them through this Google Drive link:

https://drive.google.com/file/d/1MIF3huKuSJMiQhxfqeTeCya25P8dh16x/view?usp=sharing

**Please put this file into ./analyses folder and decompress it!**

In ./analyses, we provide the source code of our experiemnts (figxx.py), auxilliary data (./analyses/cases for covid cases curve, ./analyses/population for the number of citizens, ./analyses/fig1c_data for Apple Mobility Trends data, ./analyses/fig2b_data for NYC fine-grained infection), and the simulation provided by our model (./analyses/results, decompressed from the Google Drive archive).

As an example, we can reproduce fig.1a through the following procedures:


1. Activate the environment
``` bash
conda activate covid
```
2. Go to the corresponding directory
``` bash
cd ./analyses
```
3. Download the model results and decompress
``` bash
unzip results.zip
```
3. Run the following command
``` bash
python fig1a.py
```

