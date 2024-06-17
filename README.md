# How to run the code?

## Onset Detection

For the onset detection you should use the **onset_test.ipynb**. To ensure that it will run correctly 
please place the test audio files into **/data/onset/test**. Please also make sure that you have in the same folder 
the computed model with the name **best_model.pth** and the train dataset comptued **mean_std.pkl** file. 
To be able to use the utility functions you should also have a file called **utils.py** in the same folder.
If all of the above are true then it should work and predict the onsets.

## Tempo Detection

For the tempo detection you should use the **tempo_test.ipynb**. To ensure that it will run correctly
please place the test audio files into **/data/tempo/test**. Please also make sure that you have in the same folder
the computed model with the name **best_model.pth** and the train dataset comptued **mean_std.pkl** file.
To be able to use the utility functions you should also have a file called **utils.py** in the same folder.
If all of the above are true then it should work and predict the tempos.

