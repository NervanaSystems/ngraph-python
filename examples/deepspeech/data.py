from __future__ import division
import numpy as np
from ngraph.frontends.neon.aeon_shim import AeonDataLoader
from ngraph.util.persist import get_data_cache_or_nothing


class SpeechTranscriptionLoader(AeonDataLoader):
    def __new__(cls, manifest_filename, audio_length, transcript_length,
                sample_freq_hz=16000, frame_length=.025, frame_stride=.01,
                feature_type='mfsc', num_filters=13,
                alphabet="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
                batch_size=32, cache_root=None, num_batches=None,
                single_iteration=False, seed=None):
        """
        Custom dataloader for speech transcription.

        Arguments:
            manifest_filename (str): Path to manifest file
            audio_length (float): Length of longest audio clip (seconds)
            transcript_length (int): Length of longest transcription
            sample_freq_hz (int): Sample rate of audio files (hertz)
            frame_length (float): Length of window for spectrogram calculation (seconds)
            frame_stride (float): Stride for spectrogram calculation (seconds)
            feature_type (str): Feature space for audio
            num_filters (int): Number of mel-frequency bands
            alphabet (str): Alphabet for the character map
            batch_size (int): Size of a single batch
            cache_root (str): Path to dataloader cache directory
            num_batches (int): Number of batches to load. Defaults to infinite
            single_iteration (bool): Sets "iteration_mode" to "ONCE"
            seed (int): Random seed for dataloader. Also turns off shuffling.
        """

        if cache_root is None:
            cache_root = get_data_cache_or_nothing('deepspeech2-cache/')

        feats_config = dict(type="audio",
                            sample_freq_hz=sample_freq_hz,
                            max_duration="{} seconds".format(audio_length),
                            frame_length="{} seconds".format(frame_length),
                            frame_stride="{} seconds".format(frame_stride),
                            feature_type=feature_type,
                            num_filters=num_filters,
                            emit_length=True)

        # Transcript transformation parameters
        transcripts_config = dict(type="char_map",
                                  alphabet=alphabet,
                                  max_length=transcript_length,
                                  emit_length=True)

        config = {'manifest_filename': manifest_filename,
                  'batch_size': batch_size,
                  'etl': (feats_config, transcripts_config),
                  'cache_directory': cache_root}

        if seed is not None:
            config["shuffle_enable"] = False
            config["shuffle_manifest"] = False
            config["random_seed"] = seed

        if num_batches is not None:
            config["iteration_mode"] = "COUNT"
            config["iteration_mode_count"] = num_batches
        elif single_iteration is True:
            config["iteration_mode"] = "ONCE"

        return super(SpeechTranscriptionLoader, cls).__new__(cls, config)

    def next(self):

        sample = super(SpeechTranscriptionLoader, self).next()
        return self._preprocess(sample)

    def _preprocess(self, sample):
        """
        Preprocess samples to pack char_map for ctc, ensure dtypes,
        and convert audio length to percent of max.

        Arguments:
            sample (dict): A single sample dictionary with keys of
                           "audio", "audio_length", "char_map", and
                           "char_map_length"
        """

        sr = self.config["etl"][0]["sample_freq_hz"]
        dur = float(self.config["etl"][0]["max_duration"].split()[0])

        def pack_for_ctc(arr, trans_lens):

            packed = np.zeros(np.prod(arr.shape), dtype=arr.dtype)
            start = 0
            for ii, trans_len in enumerate(trans_lens):
                packed[start: start + trans_len] = arr[ii, 0, :trans_len]
                start += trans_len

            return np.reshape(packed, arr.shape)

        sample["audio_length"] = 100 * sample["audio_length"].astype("float32") / (sr * dur)
        sample["audio_length"] = np.clip(sample["audio_length"], 0, 100).astype(np.int32)
        sample["char_map"] = pack_for_ctc(sample["char_map"],
                                          sample["char_map_length"].ravel()).astype(np.int32)
        sample["char_map_length"] = sample["char_map_length"].astype(np.int32)
        sample["audio"] = sample["audio"].astype(np.float32)

        return sample
