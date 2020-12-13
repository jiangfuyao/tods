import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.KDiscordODetect import KDiscordODetectorPrimitive

class KDiscordODetectorSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=KDiscordODetectorPrimitive, **hyperparams)

