import numpy as np
import torch
import mne
import os, sys
sys.path.append(".")
sys.path.append("./code/")
from code.eeg_recording import SingleSubjectRecording
import scipy


raw_data_folder = '../data/ethz_ieeg/long-term/ID01/'

EEG_raw_files = os.listdir(raw_data_folder)
example_data = scipy.io.loadmat(os.path.join(raw_data_folder, EEG_raw_files[0]))['EEG']
mne_example_data = mne.io.RawArray(example_data, mne.create_info(ch_names=[f'eeg unknow_ch_{i}'for i in range(example_data.shape[0])], sfreq=512))

mne_example_data.set_channel_types(dict(zip(mne_example_data.ch_names, ['eeg'] * len(mne_example_data.ch_names))))

recording = SingleSubjectRecording(0 << 8 + 2, mne_example_data)
recording.run_latent_hmm(n_states=9, use_gfp=True)
np.save("./microstate_maps.npy", recording.latent_maps)
print(recording.gev_tot)
