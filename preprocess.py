# preprocess.py
from music21 import converter, note, chord
import os
import pickle

def process_midi_folder(dataset_path):
    """
    Loops through all MIDI files in a folder and extracts the notes/chords.
    """
    all_notes = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".mid", ".midi")):
                file_path = os.path.join(root, file)
                try:
                    midi_stream = converter.parse(file_path)
                    notes_to_parse = midi_stream.flatten().notes
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            all_notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            all_notes.append('.'.join(str(n) for n in element.normalOrder))
                except Exception as e:
                    print(f"  Could not process file {file_path}. Error: {e}")
    return all_notes

def prepare_sequences(notes, sequence_length=30):
    """
    Prepares sequences for the neural network.
    """
    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    network_input = [[note / float(n_vocab) for note in sequence] for sequence in network_input]
    
    return (network_input, network_output, n_vocab)