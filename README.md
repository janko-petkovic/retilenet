# RetiLeNet (retinet)
This is the code used to generate the data for the RetiLeNet submission.

Note to the reader: *Unfortunately, the final name was chosen after the whole study was completed using the
prototype name ```retinet``` throughout the whole code. Changing it would have represented a major
bug source, and with the incoming deadline chose to keep the code as it was and correct only the final outputs. This
issue will, however, be addressed in the future*.

# Usage
## Prerequisites
The scripts are based on the well known python and pytorch libraries:
- torch
- torchvision
- tqdm
- matplotlib
- numpy
- pillow

You can install them using PIP
```
pip install torch torchvision tqdm matplotlib numpy pillow
```

or conda (you have to verify on which channel the repositories are located in this case though).


## Running
After cloning the repo in your folder of choice 
```
cd folder_of_choice
git clone https://github.com/janko-petkovic/retinet.git
```
you will find two types of scripts in the root folder.


### Notebooks: training, accuracy sweep
As the name suggests these are scripts used for training and comparing the final accuracies on modified datasets:
- ```training.ipynb```: train a specific model (LeNet5, different types of RetiLeNet), on a specified dataset. The necessary parameters can be specified inside the ```input panel``` cell inside the notebook and the available options are provided as text comments therein.
- ```accuracy_sweep.ipynb```: once that you have trained your RetiLeNet of choice **and a LeNet5**, compare them on the desired dataset versus a $\mu$ and $\sigma$ sweep. Again, the necessary parameters can be specified inside the ```input panel``` cell in the notebook.

Being both of these notebooks GPU dependant, in case you found yourself in the lack of your favourite Titan RTX Ultimate, you can run these notebooks on Colab. In that case I warmheartedly suggest that you use this procedure:
1. In your drive create a ```Code``` folder
2. Upload the entire ```retinet``` folder into the ```Code``` folder and rename it ```RetiNet```

After that run the modules, remembering to select the correct ```Mount drive, PATH``` cell in the ```Prolegomena``` section

**Caveat:** you can upload only the notebooks and reconstruct the folder tree and change the PATH cells (everything should be coded so that those are the PATH variables are only parameters to be modified) but I don't really recommend it. I am aware though that uploading the whole folder onto drive is far from best practice and I will try to optimize this procedure in future projects.

### Scripts
The remaining scripts generate the charts included in the article. The instructions for their usage are very straightforward and should be easily retreivable from the ```-h``` option:
```
python SCRIPT_OF_CHOICE.py -h
```
Remember first to train the models and generate the data that you want to run the scripts for.


For any questions, issues or curiosities don't hesitate to contact me!
