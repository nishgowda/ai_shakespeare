# Pseudo Shakespeare (AI that writes like Shakespeare)
A tensorflow implementation of a char-rnn for character level text generation meant to emulate writing of Shakespeare.

## Training
First, download a [Shakespeare dataset](https://raw.githubusercontent.com/nishgowda/ai_shakespeare/master/data/shakespeare.txt) (or use one of your own) and place it in the *data* directory.
Run ``shakespeare.py`` along with the location of your datatset and the name of model to save.
```
 $ python3 shakespare.py data/shakespeare.txt shakespeare
```

After training the model, it should save in the *models* directory.

## Generating output
Once the model is saved, you can now run it to produce **Shakespearian** writing!
Run ``genereate.py`` along with then name of the previously saved model in the models directory.
```
  $ python3 generate.py shakespeare
```

This will save the output in the *output* directory and print the name of the file.

 

