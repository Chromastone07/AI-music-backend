from music21 import converter, instrument, note, chord

def inspect_midi_file(file_path):
    """
    Opens a MIDI file and prints the musical elements it finds.
    """
    print(f"--- Inspecting File: {file_path} ---")
    
    try:
        # Load the MIDI file
        midi_stream = converter.parse(file_path)
        
        # We need to flatten the stream to get all notes from all parts/instruments
        notes_to_parse = midi_stream.flat.notes
        
        print(f"Found {len(notes_to_parse)} total notes, chords, and rests.")
        
        # Loop through the first 20 elements to see what we have
        for i, element in enumerate(notes_to_parse):
            if i >= 20: # Stop after 20 elements to keep the output clean
                break
                
            if isinstance(element, note.Note):
                print(f"  Note: {element.pitch}, Duration: {element.duration.type}")
            elif isinstance(element, chord.Chord):
                # A chord is a set of notes played at the same time
                # We get the normal order of pitches (e.g., [C4, E4, G4])
                notes_in_chord = '.'.join(str(n) for n in element.normalOrder)
                print(f"  Chord: {notes_in_chord}, Duration: {element.duration.type}")
            elif isinstance(element, note.Rest):
                print(f"  Rest, Duration: {element.duration.type}")

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

# --- Your Task ---
if __name__ == "__main__":
    # Replace 'your_file.mid' with the actual name of your MIDI file
    your_midi_file = 'Pirates of the Caribbean - He\'s a Pirate (1).mid' 
    inspect_midi_file(your_midi_file)