import librosa
import numpy as np

from docupilot.recording.session import RecordingSession

class AudioFeatureExtractor:
    """
    Provides audio feature extraction.
    """

    @staticmethod
    def extract_audio_features(recording_session: RecordingSession):
        """
        Extract the audio features from the recording session.
        :param recording_session: The recording session contains the path to the mp4 file.
        :return: None
        """

        # Load the audio file
        audio, sampling_rate = librosa.load(recording_session.recording_path)

        # Extract the mfcc features and delta features
        mfcc = librosa.effects.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
        delta = librosa.effects.feature.delta(mfcc)
        delta2 = librosa.effects.feature.delta(mfcc, order=2)

        # Extract the rms features
        rms = librosa.effects.feature.rms(y=audio)

        # Build a feature vector containing the mfcc, delta, delta2, and rms features
        combined_audio_features = np.vstack([mfcc, delta, delta2, rms])

        return combined_audio_features.T


class VideoFeatureExtractor:
    """
    Provides video feature extraction.
    """

    def __init__(self):
        """
        Initializes the feature extractor.
        """

    def extract_video_features(self, recording_session: RecordingSession) -> None:
        pass


class EventFeatureExtractor:
    """
    Provides event feature extraction.
    """

    def __init__(self):
        """
        Initializes the feature extractor.
        """

    def extract_event_features(self, recording_session: RecordingSession) -> None:
        pass
