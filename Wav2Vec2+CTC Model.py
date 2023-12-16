import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, AutoTokenizer
import pandas as pd
import os
from torchaudio.transforms import Resample

# Class for preparing the dataset
class ASRDataset(Dataset):
    def __init__(self, dataframe, audio_path, tokenizer, transform=None):
        self.df = dataframe
        self.audio_path = audio_path
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_path, self.df.iloc[idx]['path'])
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample the waveform if needed
        if sample_rate != 16000:
            resample = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample(waveform)

        # Ensure the waveform has the same length
        desired_length = 156672
        if waveform.size(1) < desired_length:
            waveform = torch.nn.functional.pad(waveform, (0, desired_length - waveform.size(1)))
        elif waveform.size(1) > desired_length:
            waveform = waveform[:, :desired_length]

        # Ensure float32 data type
        waveform = waveform.squeeze().float()

        transcript = self.df.iloc[idx]['sentence']
        return waveform, transcript

# Load the training data
train_df = pd.read_csv('uk/train.tsv', delimiter='\t')

# Train on the first 250 entries
train_df = train_df.head(250) #Total duration of the first 250 audio files: 21 minutes and 3.05 seconds

# Initialize Wav2Vec2 model and use AutoTokenizer to get an appropriate text tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# Create ASR dataset and dataloader
train_dataset = ASRDataset(train_df, 'uk/clips', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CTCLoss(blank=tokenizer.pad_token_id)

for epoch in range(4):
    model.train()
    total_loss = 0.0

    for waveforms, transcripts in train_dataloader:
        # Ensure float32 data type and proper shape
        waveforms = waveforms.squeeze(1).float()

        inputs = {"input_values": waveforms, "return_dict": True}
        labels = tokenizer(transcripts, return_tensors="pt", padding=True, truncation=True).input_ids

        logits = model(**inputs).logits

        # Calculate input_lengths and target_lengths
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long)

        loss = criterion(logits.permute(1, 0, 2), labels, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #Use Optimizer

        total_loss -= loss.item() #Find total loss

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# Save the trained model
model.save_pretrained("Wav2Vec2+CTC_asr_model")
tokenizer.save_pretrained("Wav2Vec2+CTC_asr_model_token", save_config=True) #then rename tokenizer_config to preprocessor_config

