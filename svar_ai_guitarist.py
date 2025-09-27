# ==============================================================================
# CELL 2: Create the Python Script
# ==============================================================================
import glob
import pickle
import numpy as np
import argparse
import os
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Define File Paths ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/SvarAI"
DATA_DIR = os.path.join(DRIVE_BASE_PATH, "data")
NOTES_PATH = os.path.join(DATA_DIR, "notes")
WEIGHTS_PATH = os.path.join(DRIVE_BASE_PATH, "weights-guitar-lead-best.hdf5")
OUTPUT_PATH = os.path.join(DRIVE_BASE_PATH, "svar_ai_composition.mid")

def parse_midi_files():
    print("--- Parsing MIDI Files from Kaggle Hub Dataset ---")
    
    try:
        with open(os.path.join(DATA_DIR, 'dataset_path.txt'), 'r') as f:
            dataset_path = f.read().strip()
    except FileNotFoundError:
        print("Error: dataset_path.txt not found. Please run the setup notebook cell first.")
        return None
    
    midi_search_path = os.path.join(dataset_path, "**/*.mid")
    notes = []
    
    file_list = glob.glob(midi_search_path, recursive=True)
    print(f"Found {len(file_list)} MIDI files to parse.")

    for file in file_list:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            for part in parts:
                if 'Guitar' in str(part.getInstrument()):
                    notes_to_parse = part.recurse()
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))
        except Exception:
            pass
            
    with open(NOTES_PATH, 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    print(f"--- Parsing Complete. Found {len(notes)} guitar notes. ---")
    return notes

def prepare_sequences_for_training(notes, sequence_length=32):
    print("--- Preparing Sequences for Training ---")
    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)
    
    with open(os.path.join(DATA_DIR, 'pitchnames'), 'wb') as filepath:
        pickle.dump(pitchnames, filepath)
        
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)
    print("--- Sequence Preparation Complete ---")
    return network_input, network_output, n_vocab

def create_network(network_input, n_vocab):
    print("--- Creating LSTM Model Architecture ---")
    model = Sequential([
        LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True),
        LSTM(512, return_sequences=True, recurrent_dropout=0.3),
        LSTM(512),
        BatchNorm(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNorm(),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("--- Model Created ---")
    model.summary()
    return model

def train_network(model, network_input, network_output, epochs=100, batch_size=128):
    print("--- Starting Model Training (This will take a long time!) ---")
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    print("--- Training Complete ---")

def generate_music():
    print("--- Generating Music ---")
    with open(NOTES_PATH, 'rb') as filepath:
        notes = pickle.load(filepath)
    with open(os.path.join(DATA_DIR, 'pitchnames'), 'rb') as filepath:
        pitchnames = pickle.load(filepath)
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    sequence_length = 32
    network_input = []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []
    model = load_model(WEIGHTS_PATH)
    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    create_midi_file(prediction_output)

def create_midi_file(prediction_output):
    print("--- Converting to MIDI File ---")
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.ElectricGuitar()
        output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=OUTPUT_PATH)
    print(f"--- Composition saved to: {OUTPUT_PATH} ---")

def main():
    parser = argparse.ArgumentParser(description="Svar.ai - AI Guitarist")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'generate'],
                        help="Set the mode to 'train' a new model or 'generate' music.")
    args = parser.parse_args()

    if args.mode == 'train':
        notes_data = parse_midi_files()
        if not notes_data:
            print("\n\nERROR: No guitar notes were found in the dataset.")
            print("This could be because the instrument names in the MIDI files are different.")
            print("Try checking the `part.getInstrument()` names inside the script.\n")
            return
        network_in, network_out, n_vocab = prepare_sequences_for_training(notes_data)
        model = create_network(network_in, n_vocab)
        train_network(model, network_in, network_out)
    elif args.mode == 'generate':
        generate_music()

if __name__ == "__main__":
    main()
