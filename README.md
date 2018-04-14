# DeepHadTopTagger
**A deep neural network for tagging triplets of jets from hadronic top decay.**

## Pipeline
----
- ### Generate Signal & Background:
  - Run `root -l makeFlatTreesForMatt.C+` from within the CMSSW environment on Earth.crc.nd.edu.
- ### Create Dataset:
  - Open `python` interactively.
  - Create a `utils.CollisionDataset` object for each signal & background file that you are interested in.
  - I recommend taking the size of the smallest dataset and subsampling all of the other datasets to the same size.
  - Add the datasets together.
  - Use `utils.CollisionDataset.saveas(<filename>)` to save the dataset as a `.npy` for loading again later.
- ### Train the Neural Network:
  - Run `python train_nn.py`.
  - The model should be saved at `neural_net.torch`.
- ### Evaluate the Model
  - View `model_analysis.ipynb` for help here.
  
