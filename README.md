# Some-R-Set Training

This project contains some code that attempts to use machine learning to produce a trained model 
to play the game [Some-R-Set](https://www.somersetgame.com).  This game was chosen because it is 
a realitvely unknown game that uses a non-standard deck of cards. 

## Contents

### Local Training

Python [code](local/README.md) that can be executed to train a model on your own hardware is provided. 

### Notebook

Depending on your hardware, you might have trouble running the training code locally since some of the libraries are very hardware-specific.  The code has been transposed into an [interactive Python notebook](notebook/README.md), which can be used in cloud-based platforms like Google Colab that will be more likely to have libraries tailored to the hardware. Note that this will utilize your personal Google Drive for storage of the model and logs. 