# Pseudo Shakespeare (AI that writes like Shakespeare)
A tensorflow implementation of a char-rnn for character level text generation meant to emulate writing of Shakespeare.

## Training
First, download a [Shakespeare dataset](https://raw.githubusercontent.com/nishgowda/ai_shakespeare/master/data/shakespeare.txt) (or use one of your own) and place it in the *data* directory.
Run ``shakespeare.py``
```
 $ python3 shakespare.py
```

After training the model, it should save in the *output* directory.

## Generating output
Once the model is saved, you can now run it to produce **Shakespearian** writing!
Run ``genereate.py`` along with the location of the previously saved model.
```
  $ python3 generate.py models/shakespeare
```

 

