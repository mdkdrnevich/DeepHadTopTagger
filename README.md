# DeepHadTopTagger
**Deep neural network research for tagging triplets of jets from hadronic top decay for the CMS experiment at CERN.**

----
## Overview
A challenging problem that is of current interest at CERN is to determine whether or not (i.e. binary classification) three jets of quarks found in a given event (particle collision) are descendents from a top quark produced in the event. Typically, if a top quark is produced, the top will decay into a bottom quark and W boson, then the W boson might decay into two hadrons, and the bottom quark plus the two additional hadrons would be detected by CMS to form our triplet of interest.

Current models in use for this classification problem are various versions of boosted decision trees. These BDTs use engineered features about the particles to perform the classification. Our approach is to eliminate the need for engineered features since one never knows when you have found enough such variables and can cost researchers a significant amount of their time. Instead, assuming some weak conditions, the universal approximation theorem states that neural networks should be able to use just the features constructed from detector-level information to reach the theoretical optimum. Thus we are developing neural networks that outperform current BDTs to try and reach this optimum. This serves as both an improved ''hadronic top decay tagger'' and a testament to the plug and play power of neural networks with only minimal feature engineering.

We have achieved a 7.44% relative improvement in accuracy (defined in the results section) over the best BDT using the neural network described below. 

## Pipeline
- ### Generate Signal & Background:
  - _Note that all of the ROOT files assume use of the CMSSW environment for certain object definitions_
  - From within the DataGeneration folder
    - makeFlatTreesForMatt.C : Edit the function block at the end as necessary and run in ROOT to generate csv files of signal and background data.
    - makeTagTripletsSet.C : Similar to makeFlatTreesForMatt.C except that the label written for the event is a 3-tuple indicating the indices of the hadronic top decay triplet (filters on events having exactly one such decay).
- ### Create Dataset:
  - Look at `preprocessing.py` for inspiration.
    - I use my custom `utils.CollisionDataset` class to make data preprocessing significantly easier.
- ### Train the Neural Network:
  - Run `python train_nn.py`.
  - The model should be saved as `neural_net.torch`.
- ### Evaluate the Model
  - View `model_analysis.ipynb` for help here.
  
## Results

### Model

### Statistics
