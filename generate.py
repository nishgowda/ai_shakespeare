import random
import sys
import tensorflow as tf
import os
import numpy as np

text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char_to_index = {u:i for i, u in enumerate(vocab)}
index_to_char = np.array(vocab)
# generate text from model with a word to start off with

def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 0.1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions /= temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])
    return (start_string + ''.join(text_generated))

# write the contents of generate_text to a file
def write_to_file(output, filename):
    fi = f"output/{filename}.txt"
    with open(fi, "w") as f:
        f.write(output)
    print("Succesfully wrote pseudo shakespeare to the file ", fi)

if __name__ == "__main__":
    model_name = sys.argv[1]
    if model_name == None:
        sys.exit("Must include model name in models directory")
    model = tf.keras.models.load_model(f'models/{model_name}')
    print("Loaded model...")
    model.summary()
    options = ["ROMEO", "JULIET", "EDMUND", "KING_HENRY", "OTHELLO", "MACBETH"]
    character = sys.argv[2]
    if character = None:
        character = random.choice(options)
    start_string = character + ":"
    gen_text = generate_text(model, start_string=start_string)
    write_to_file(gen_text, character)
