# DeepHadTopTagger
**A deep neural network for tagging triplets of jets from hadronic top decay.**

## Pipeline
----
- ### Generate Signal & Background:
  - Run `root -l makeFlatTreesForMatt.C+` from within the CMSSW environment on Earth.crc.nd.edu.
- ### Create Dataset:
  - Look at `preprocessing.py` for inspiration.
    - I use my custom `utils.CollisionDataset` class to make data preprocessing significantly easier.
- ### Train the Neural Network:
  - Run `python train_nn.py`.
  - The model should be saved as `neural_net.torch`.
- ### Evaluate the Model
  - View `model_analysis.ipynb` for help here.
  
