import mido
import numpy as np
from icecream import ic

class EventTokenizer:
    """Encodes MIDI files to event-based representation.
    Decodes event vectors back to MIDI (with default tempo and time-signature)"""
    
    end_token = -1
    nb_waits = 1000

    #data augmentation parameters
    nb_transpose_above = 12
    nb_transpose_below = 12

    #for writing MIDI only
    velocity = 64
    write_tempo = 800000

    def __init__(self, nb_voices) -> None:
        self.nb_voices = nb_voices
        self.vocab_size = 0
 
        self.events = (
            ['start', 'end'] +
            ['wait_'+str(i) for i in range(EventTokenizer.nb_waits)] +
            ['note_on_'+str(i) for i in range(21, 109)] + #21-108 midi pitches for entire keyboard
            ['note_off_'+str(i) for i in range(21, 109)] + 
            ['switch_to_voice_'+str(i) for i in range(nb_voices)]  #first track of metadata doesn't count
        )
        self.vocab_size = len(self.events)

        self.event_map = {event:idx for idx,event in enumerate(self.events)}
        #self.event_map['end'] = EventTokenizer.end_token #might not be a good idea

        self.inv_event_map = None #lazy instanciation
        
  
    def file_to_vector(self, filepath: str, augment=False, return_mask=False) -> np.ndarray:    
        return self.midi_to_vector(mido.MidiFile(filepath), augment=augment, return_mask=return_mask)


    def _encode(self, messages: mido.midifiles.tracks.MidiTrack, return_mask: bool = False) -> np.ndarray:
        #clocks = 0
        current_channel = 0
        current_note = [None for _ in range(self.nb_voices)]
        vector = [self.event_map['start']]
        for i,message in enumerate(messages):
            if isinstance(message, mido.MetaMessage):
                if message.type == 'end_of_track': 
                    vector.append(self.event_map['end'])
                    break 
                continue
            if message.channel != current_channel:
                vector.append(self.event_map['switch_to_voice_'+str(message.channel)])
                current_channel = message.channel
            if message.time > 0:
                vector.append(self.event_map['wait_'+str(message.time)])
                #clock += message.time
            if message.type == 'note_on':
                vector.append(self.event_map['note_on_'+str(message.note)])
                current_note[current_channel - 1] = message.note
            elif message.type == 'note_off':
                vector.append(self.event_map['note_off_'+str(message.note)])
                if message.note != current_note[current_channel - 1]:
                    print(f"wrong note...iteration {i}")
                current_note[current_channel - 1] = None
            else:
                print(f"Unknown message type: {message.type}")
        if vector[-1] != self.event_map['end']:
            print("It never ended...")
            vector.append(self.event_map['end'])
        print(f"Final iteration {i}")
        if return_mask:
            return vector, [1 for _ in range(len(vector))]
        return vector
                 

    def _encode_and_augment(self, messages: mido.midifiles.tracks.MidiTrack, return_mask=False) -> np.ndarray:
        #clocks = [0 for _ in range(self.nb_voices)]
        current_channel = 0
        vector = [[self.event_map['start']] for _ in range(12)]

        started = False

        for message in messages:
            if isinstance(message, mido.MetaMessage):
                if message.type == 'end_of_track':
                    for row in vector:
                        row.append(self.event_map['end'])
                    break 
                continue
            
            if message.channel != current_channel or not started:
                for row in vector:
                    row.append(self.event_map['switch_to_voice_'+str(message.channel)])
                current_channel = message.channel
                started = True
            if message.time > 0:
                for row in vector:
                    row.append(self.event_map['wait_'+str(message.time)])
            if message.type == 'note_on':
                for augmentation_offset, row in zip(range(-EventTokenizer.nb_transpose_below, EventTokenizer.nb_transpose_above + 1), vector):
                    row.append(self.event_map['note_on_'+str(message.note + augmentation_offset)])
            elif message.type == 'note_off':
                for augmentation_offset, row in zip(range(-EventTokenizer.nb_transpose_below, EventTokenizer.nb_transpose_above + 1), vector):
                    row.append(self.event_map['note_off_'+str(message.note + augmentation_offset)])
                # if message.note != current_note[current_channel - 1]:
                #     print("wrong note...")
                # current_note[current_channel - 1] = None
            else:
                print(f"Unknown message type: {message.type}")
        if vector[1][-1] != self.event_map['end']:
            print("It never ended...")
            vector.append(self.event_map['end'])
        if return_mask:
            return vector, [1 for _ in range(len(vector))]
        return vector
        

    def midi_to_vector(self, mid: mido.midifiles.midifiles.MidiFile, augment: bool = False, return_mask: bool = False) -> np.ndarray:
        if len(mid.tracks) - 1 != self.nb_voices:
            raise Exception(f"Inconsistent number of voices: tokenizer has {self.nb_voices}, MIDI has {len(mid.tracks) - 1}")
        #print(mid.__dict__.keys())
        #print(mid.__dir__())
        #merged = mid.merge_tracks(mid.tracks, skip_checks=True)
        if augment:
            return self._encode_and_augment(mid.merged_track, return_mask=return_mask)
        else:
            return self._encode(mid.merged_track, return_mask=return_mask)
            

    def vector_to_midi(self, vector: np.ndarray, output_filepath: str = None) -> mido.midifiles.midifiles.MidiFile:
        if self.inv_event_map is None:
            self.inv_event_map = {v:k for k,v in self.event_map.items()}
        mid = mido.MidiFile()

        #just to put something down for metadata
        metatrack = mido.MidiTrack([
            mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0),
            mido.MetaMessage('set_tempo', tempo=EventTokenizer.write_tempo, time=0),
            mido.MetaMessage('end_of_track', time=0)
        ])
        mid.tracks.append(metatrack)
        tracks = [mido.MidiTrack() for _ in range(self.nb_voices)]
        for i, track in enumerate(tracks):
            track.append(mido.MetaMessage('text', 'voice_'+str(i), time=0))
            mid.tracks.append(track)
        
        current_channel = 0
        clocks = [0 for _ in range(self.nb_voices)]
        #started = [False for _ in range(self.nb_voices)]
        global_clock = 0
        for event_code in vector.flatten():
            event = self.inv_event_map[event_code]
            if event.startswith("e"):
                for track in tracks:
                    track.append(mido.MetaMessage('end_of_track', time=0))
            elif event.startswith("start"):
                continue
            elif event.startswith("s"):
                current_channel = int(event.split("_")[-1])
            elif event.startswith("w"):
                wait_time =  int(event.split("_")[-1])
                global_clock += wait_time
            elif event.startswith("note_on"):
                current_note = int(event.split("_")[-1])
                tracks[current_channel].append(
                    mido.Message('note_on', 
                                 channel=current_channel, 
                                 note=current_note, 
                                 velocity=EventTokenizer.velocity, 
                                 time=global_clock - clocks[current_channel])
                )
                clocks[current_channel] = global_clock
            elif event.startswith("note_off"):
                current_note = int(event.split("_")[-1])
                tracks[current_channel].append(
                    mido.Message('note_off', 
                                 channel=current_channel, 
                                 note=current_note, 
                                 velocity=EventTokenizer.velocity, 
                                 time=global_clock - clocks[current_channel])
                )
                clocks[current_channel] = global_clock
            else:
                print("unknown event...")
        if output_filepath is not None:
            mid.save(filename=output_filepath)
        return mid

    


        


