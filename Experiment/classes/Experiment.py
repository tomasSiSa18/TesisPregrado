from Experiment.classes import Encoder


class Experiment:
    
    def __init__(self, device, dataset, encoder: Encoder):
        self.device = device
        self.dataset = dataset
        self.encoder = encoder
    
        
    
        
    