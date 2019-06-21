#autism
class Diagnosis_class:
    speaker_ID = None
    features = None
    label = None

    def __init__(self,label,speaker_ID,features):
        self.speaker_ID = speaker_ID
        self.label = label
        self.features = features