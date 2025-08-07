# generate.py (Corrected for Local Demo)
import numpy as np
from music21 import instrument, note, stream, chord

# We no longer need pickle or train_model here

def generate_notes(model, notes, n_vocab, pitchnames):
    """ Generate notes from the pre-trained neural network. """
    
    # We get pitchnames as an argument now, so we don't need to recalculate it.
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Create a random starting sequence
    sequence_length = 30
    start = np.random.randint(0, len(notes) - sequence_length - 1)
    pattern = notes[start:start + sequence_length]
    int_pattern = [note_to_int[note_str] for note_str in pattern]

    prediction_output = []
    
    # Generate 300 notes
    for _ in range(300):
        prediction_input = np.reshape(int_pattern, (1, len(int_pattern), 1))
        # Normalize the input
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        
        index = np.argmax(prediction)
        result = pitchnames[index]
        prediction_output.append(result)

        int_pattern.append(index)
        int_pattern = int_pattern[1:len(int_pattern)]

    return prediction_output

def create_midi(prediction_output, filename="demo_output.mid"):
    """ Convert the output from the prediction to a MIDI file. """
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            
        offset += 0.5
        
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
    return filename