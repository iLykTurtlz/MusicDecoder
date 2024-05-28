from event_tokenizer import *
from icecream import ic
import os


# Prepare the dataset
tok = EventTokenizer(2)
vector = tok.file_to_vector("../BachTwoPartInventions/invent1.mid")

ic(vector[:5])



# data = []
# for filename in os.scandir("../BachTwoPartInventions/"):
#     if filename.is_file():
#         songs = tok.file_to_vector(filename, augment=True, return_mask=False)
#         data += songs

# max_length = max(len(song) for song in data)
# ic(max_length)






# #vectors = np.asarray(vectors)
# ic(max(len(vector) for vector in vectors))


#tok.vector_to_midi(vector, "./test_tokenizer.mid")


print("Done")