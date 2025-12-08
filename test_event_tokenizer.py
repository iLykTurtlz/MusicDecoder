from event_tokenizer import *
from icecream import ic
import os
import random

NB_VOICES = 5

# Prepare the dataset
tok = EventTokenizer(NB_VOICES)
#vector = tok.file_to_vector("../BachTwoPartInventions/invent1.mid")

#ic(vector[:5])



data = []
for filename in os.scandir("../BachTwoPartInventions/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs

for filename in os.scandir("../BachThreePartInventions/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs

for filename in os.scandir("../BachFugues/"):
    if filename.is_file():
        songs = tok.file_to_vector(filename, augment=True, return_mask=False)
        data += songs


ic(len(data))
ic(max(len(song) for song in data))

# max_length = max(len(song) for song in data)
# ic(max_length)





vector = data[random.randint(0, len(data)-1)]
ic(vector[:30])



# #vectors = np.asarray(vectors)
# ic(max(len(vector) for vector in vectors))


tok.vector_to_midi(vector, "./test_a3.mid")
ic([tok.inv_event_map[token] for token in vector[:30]])

print("Done")