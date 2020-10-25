# MoA Experimentation and Development Pipeline
## Setup
Download the MoA datasets and put them in input

Run load_data() in run.py to generate csv's we'll use for training

## Experimenting
Tweak the model and learning paramaters in models/neural_net.py

Run run_experiment() in run.py to test the model and save

## To-Dos
- Model-loading and prediction
- Multi-label stratified k-fold
- Find out how to save Keras architecture's so that they're editable after load