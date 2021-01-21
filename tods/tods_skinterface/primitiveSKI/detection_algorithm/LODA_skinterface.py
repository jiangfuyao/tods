import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodLODA import LODAPrimitive

class LODASKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LODAPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
