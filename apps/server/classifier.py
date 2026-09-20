import numpy as np

class BirdClassifier:
    def __init__(self):
        print("Bird classifier placeholder initialized")

    def predict(self, audio, sample_rate=44100):
        # TODO: Replace this method with your trained bird classifier.
        if audio is None or len(audio)==0:
            return {"species":"No audio","confidence":0.0,"detected":False}
        rms=float(np.sqrt(np.mean(np.square(audio))))
        if rms>0.01:
            return {"species":"Bird detected (placeholder)","confidence":0.85,"detected":True}
        return {"species":"No bird detected","confidence":0.0,"detected":False}
