from transformer import *
from datetime import datetime
from event_tokenizer import *
from icecream import ic
import torch
import torch.nn.functional as F

#vocab_size = 2179
#max_length = 3873

#model = Decoder(d_k=16, d_v=16, d_model=16, nb_heads=8, nb_layers=3, dropout_proba=0.1, max_len=max_length, vocab_size=vocab_size)


NB_VOICES = 3
SOFTMAX = False
TEMPERATURE = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#old model trained on 2-3 part inventions
load_path="/datasets/pjarski/model_with_args_2024-05-31 18:32:35.pth"

#new model trained 200 iterations on all data
#load_path='/datasets/pjarski/model_with_args_2024-06-02 13:15:23.pth'

checkpoint = torch.load(load_path)
model_state_dict = checkpoint['model_state_dict']
init_args = checkpoint['init_args']
model = Decoder(**init_args)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

start_gen = datetime.now()


# Fugue prompt and mask
prompt_list = [0, 2179, 122, 2049, 62, 2137, 2048, 62, 2136, 2049, 122, 2137, 2044, 122, 2132, 2045, 122, 2133, 2049, 62, 2048, 5, 2137, 59, 2136, 2049, 122, 2137, 2051, 122]
prompt = torch.tensor([prompt_list]).to(device)
mask = torch.tensor([[1 for _ in range(30)]]).to(device)





# Prompt and mask
#prompt_list = [0,2178,2044,2032,122,2120,2036,122,2124,2034,52,2132,12,2044,62,2132,2043,2122,2032,62,2131,2041,62,2129,2039,2120,2031,122,2127,2041]
#prompt = torch.tensor([prompt_list]).to(device)
#mask = torch.tensor([[1 for _ in range(30)]]).to(device)


# Empty prompt and mask
#prompt = torch.tensor([[0]]).to(device)
#mask = torch.tensor([[1]]).to(device)

ic(prompt.shape)
ic(mask.shape)

#prompt = torch.tensor([[61, 2040, 61, 2128]]).to(device)
#mask = torch.tensor([[1,1,1,1]]).to(device)
for i in range(2000):
    #ic(i)
    outputs = model(prompt, mask)

    if SOFTMAX:
        probas = F.softmax(outputs[:,-1,:] / TEMPERATURE, dim=1)
        prediction = torch.multinomial(probas, num_samples=1).squeeze(-1)
    else:
        prediction = torch.argmax(outputs[:,-1,:], axis=-1) #last one = predicted next
    prompt = torch.hstack((prompt, prediction.view(1, 1))).to(device)
    mask = torch.ones_like(prompt).to(device)

end_gen = datetime.now()
gen_time = end_gen - start_gen
ic(gen_time)




prompt = np.array(prompt.cpu()).flatten()
# ic(prompt)
# adjusted_prompt = list([i+2 for i in prompt])
ic(prompt.shape)

final = [0]
for i in prompt:
    final.append(i)
final.append(1)

print(final)


# adjusted_prompt = [0] + adjusted_prompt + [1]
# ic(adjusted_prompt)
# ic(len(adjusted_prompt))

# result = np.array(adjusted_prompt)

datetime_str = datetime.now()
tok = EventTokenizer(NB_VOICES)
tok.vector_to_midi(final, f"test_song_posttrain_{datetime_str}.mid")
