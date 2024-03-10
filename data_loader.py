import os
import pathlib
import typing as tp 
import julius
import torch
import torchaudio
from audiocraft.data.audio import audio_read
from encodec import EncodecModel
from torch.utils.data import Dataset

from fam.llm.adapeters import FlattenedInterleavedEncodec2Codebook
from fam.llm.fast_inference_utils import encode_tokens
from fam.llm.inference import SpeakerEncoder, TraindBPETokeniser, get_cached_embedding
from fam.llm.utils import normalize_text

MBD_SAMPLE_RATE = 24000
END_OF_AUDIO_TOKEN = 1024

class MetavoiceData(Dataset):
    def __init__(self, dataset_dir: str, block_size: int, validation_split: float, encodec_model: EncodecModel, tokenizer: TrainedBPETokeniser, spkemb_model: SpeakerEncoder, device: str):
        self.dataset_dir = dataset_dir
        self.block_size = block_size
        self.validation_split = validation_split
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.spkemb_model = spkemb_model
        self.device = device

        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=END_OF_AUDIO_TOKEN)

        ##loop through dataset dir and create a list of tuples (wav_path, text)
        data_list = []
        for audio_file in pathlib.Path(dataset_dir).glob('*.wav'):
            utt_id = audio_file.stem
            wav_path = f"{dataset_dir}/{utt_id}.wav"
            txt_path = f"{dataset_dir}/{utt_id}.txt"
            with open(txt_path, 'r') as f:
                text = f.read()

            wav, sr = torchaudio.load(wav_path)
            if sr != MBD_SAMPLE_RATE:
                wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)
                torchaudio.save(wav_path, wav, MBD_SAMPLE_RATE)
            
            data_list.append(wav_path, text)
        self._prepare_dataset(data_list)

        
def _prepare_dataset(self, data_list: tp.List[tp.Tuple[str, str]]):
    #we take data_list, extract all prompts and encodec tokens and append them with EOT for all of them
    # this is done to prepare the dataset for the first stage of training
    full_sequence = torch.tensor([], dtype=torch.long, device=self.device)
    spk_embds = []
    current_wavs = torch.tensor([], dtype=torch.float, device=self.device)
    current_wav_duration = 0
    for wav_path, text in data_list:
        #extract text tokenization
        prompt = self._extract_text_tokens(text)

        #extract encodec tokens
        encodec_tokens = self._extract_encodec_tokens(wav_path)

        #concatenate prompt and encodec_tokens, and EOT token at the end
        eot = torch.tensor([END_OF_AUDIO_TOKEN], dtype=torch.long, device=self.device)
        sequence = torch.cat((prompt, encodec_tokens, eot))

        #append to dataset
        full_sequence = torch.cat((full_sequence, sequence), dim=-1)

        #get wav data
        wav, sr = torchaudio.load(wav_path) #loaf the audio file
        if sr != MBD_SAMPLE_RATE:
            wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)  #average channels if stereo
        wav = wav.to(self.device)
        current_wavs = torch.cat((current_wavs, wav.unsqueeze(0)), dim=1)  #concatenate along time axis
        current_wav_duration += wav.size(0) / MBD_SAMPLE_RATE
        if current_wav_duration >= 45:
            current_wav_oath = os.path.join(self.dataset_dir, "tmp_concatenated_wavs.wav")
            torchaudio.save(current_wav_path, current_wavs.cpu(), MBD_SAMPLE_RATE)

            # extract speaker embeddings of the concatenated wav
            spk_emb = self._extract_speaker_embeddings(current_wav_path)
            spk_embds.append(spk_emb)

            #resetn
            current_wav_duration = 0
            current_wavs = torch.tensor([], dtype=torch.float32, device=self.device)
            os.remove(current_wav_path)

        #split full_sequence into training and validation

        split = int(len(full_sequence) * (1 - self.validation_split))
        self.train_dataset = full_sequence[:split]
        self.val_dataset = full_sequence[split:]

        self.spk_embds = torch.stack(spk_embds) # (N, 1, 256)

def get_batch(self, split: tp.Literal['train', 'val'], batch_size: int):
    if split == 'train':
        data = self.train_dataset
    elif split == 'val':
        data = self.val_dataset

    ix = torch.randint(0, data.size(0) - self.block_size, (batch_size,))
    x = torch.stack([data[i:i+self.block_size] for i in ix])
    y = torch.stack([data[i+1:i+se;f/block_size+1] for i in ix])

    #random batch_size number of speaker embeddings
    spk_emb = self.spk_embds[torch.randint(0, self.spk_embds.size(0), (batch_size,))] 

    return x, y, spk_emb


def _extract_text_tokens(self, text: str):
    text = normalize_text(text)
    encoded = encode_tokens(self.tokenizer, text, device=self.device)

    return encoded

def _extract_encodec_tokens(self, wav_path: str):
    #read audio
    wav, sr = audio_read(wav_path)

    #resample to MBD's expected sample rate
    if sr != MBD_SAMPLE_RATE:
        wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)

    #convert to mono and fix dimensionality
    if wav.ndim == 2:
        wav = wav.mean(axis=0, keepdims=True)
    wav = wav.unsqueeze(0)

    #extract tokens
    wav = wav.to(self.device)
    tokens = self.encodec_model.encode(wav) # list[EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]]

    tokens = tokens[0][0][0]  #(8, T)

    #only return tokens in first 2 hieararchies for training stage 1
    tokens = tokens[:2]

    tokens = tokens.flatte().to(dtype=torch.int32)
    tokens[0::2] += END_OF_AUDIO_TOKEN

    return tokens


def _extract_speaker_embeddings(self, wav_path: str):
    return get_cached_embedding(wav_path, self.spkemb_model)

