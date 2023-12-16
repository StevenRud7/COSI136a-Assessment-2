import torch
import torchaudio
import pandas as pd
import os
import string
from jiwer import wer, cer
import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample

#run pip install git+https://github.com/openai/whisper.git
# Make sure to install jiwer using: pip install jiwer
#Also make sure to install ffmpeg with conda install -c conda-forge ffmpeg and/or pip install ffmpeg

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

# Load the test data
test_df = pd.read_csv('uk/test.tsv', delimiter='\t') #Use test as baseline 
#test_df = pd.read_csv('uk/dev.tsv', delimiter='\t') #Use dev as comparison to baseline

test_df = test_df.head(250) #Total duration of the first 250 audio files: 21 minutes and 3.05 seconds

# Load the medium Whisper model
model = whisper.load_model("medium")
#model = whisper.load_model("small")

medium_language = 'uk'  # Specify the language as Ukrainian

# Load the Wav2Vec2ForCTC model and tokenizer
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("Wav2Vec2+CTC_asr_model")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("Wav2Vec2+CTC_asr_model_token")

# Create ASR dataset and dataloader
test_dataset = ASRDataset(test_df, 'uk/clips', wav2vec2_processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize variables for WER and CER calculation
audio_file_names = []
all_transcriptions = []
all_correct_transcriptions = []
correct_transcriptions_ratio = []
cer_list = []
wer_list = []

total_correct_transcriptions = 0
total_word_count_cer = 0
total_word_count_wer = 0
total_cer = 0.0
total_wer = 0.0
total_audio_files = 0


# Function to remove punctuation and convert to lowercase
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Replace Latin "i" with Cyrillic "і"
    text = text.replace("i", "і")
    return text

# Loop through the test set
for waveforms, transcripts in test_dataloader:
    # Inference with the medium Whisper model
    medium_result = model.transcribe(waveforms.numpy()[0], language=medium_language)
    medium_transcription = preprocess_text(medium_result['text'])

    # Inference with the Wav2Vec2ForCTC model and tokenizer
    inputs = wav2vec2_processor(waveforms.squeeze().numpy(), return_tensors="pt", padding=True, truncation=True, max_length=96000)
    wav2vec2_logits = wav2vec2_model(**inputs).logits
    wav2vec2_transcription = wav2vec2_processor.batch_decode(torch.argmax(wav2vec2_logits, dim=-1))[0]

    # Combine predictions (you can use voting or averaging)
    combined_transcription = medium_transcription + " " + wav2vec2_transcription

    # Apply preprocessing to the combined transcription
    combined_transcription = preprocess_text(combined_transcription)

    # Compare the combined transcription with the correct sentence
    correct_transcription = preprocess_text(transcripts[0])
    current_wer = wer(correct_transcription, combined_transcription)
    current_cer = cer(correct_transcription, combined_transcription)

    # Print results
    print(f"Combined Transcription: {combined_transcription}")
    print(f"Correct sentence: {correct_transcription}")
    print(f"Word Error Rate (WER): {current_wer}")
    print(f"Character Error Rate (CER): {current_cer}")

    # Check for any very minor typos, and if yes still consider it a correct transcription
    if current_cer <= 0.1:  # Adjust if needed
        print("Transcription is correct!\n")
        total_correct_transcriptions += 1
        total_audio_files += 1
    else:
        print("Transcription is incorrect!\n")
        total_audio_files += 1

    # Accumulate statistics for CER calculation
    total_word_count_cer += len(correct_transcription)
    total_cer += current_cer

    # Accumulate statistics for WER calculation
    total_word_count_wer += len(correct_transcription.split())
    total_wer += current_wer
    

    # Append data to lists
    audio_file_names.append(test_df.iloc[total_audio_files - 1]['path'])
    all_transcriptions.append(combined_transcription)
    all_correct_transcriptions.append(correct_transcription)
    correct_transcriptions_ratio.append(f"{total_correct_transcriptions} out of {total_audio_files}")
    cer_list.append(current_cer)
    wer_list.append(current_wer)

# Calculate the average CER
average_cer = total_cer / total_word_count_cer
print(f"Average Character Error Rate (CER): {average_cer}")

# Calculate the average WER
average_wer = total_wer / total_audio_files
print(f"Average Word Error Rate (WER): {average_wer}")

# Get Overall CER
over_cer = cer(all_correct_transcriptions, all_transcriptions)
print(f"Overall Character Error Rate (CER): {over_cer}")

# Get Overall WER
over_wer = wer(all_correct_transcriptions, all_transcriptions)
print(f"Overall Word Error Rate (WER): {over_wer}")


# Print the total number of correct transcriptions
print(f"Total correct transcriptions: {total_correct_transcriptions} out of {total_audio_files}")

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Audio File Name': audio_file_names,
    'Transcription': all_transcriptions,
    'Correct Transcription': all_correct_transcriptions,
    'Correct Transcriptions Ratio': correct_transcriptions_ratio,
    'CER': cer_list,
    'WER': wer_list
})

# Save the DataFrame to a CSV file with Ukrainian encoding
results_df.to_csv('ASR_Whisper_Wav2Vec2_Transcriptions_and_Results (Ran on test.tsv on medium whisper model).csv', index=False, encoding='utf-8-sig')
#results_df.to_csv('ASR_Whisper_Wav2Vec2_Transcriptions_and_Results (Ran on test.tsv on small whisper model).csv', index=False, encoding='utf-8-sig')
#results_df.to_csv('ASR_Whisper_Wav2Vec2_Transcriptions_and_Results (Ran on dev.tsv on medium whisper model).csv', index=False, encoding='utf-8-sig')
#results_df.to_csv('ASR_Whisper_Wav2Vec2_Transcriptions_and_Results (Ran on dev.tsv on small whisper model).csv', index=False, encoding='utf-8-sig')

