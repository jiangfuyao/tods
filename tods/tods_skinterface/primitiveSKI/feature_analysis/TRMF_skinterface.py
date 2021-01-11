import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.TRMF import TRMFPrimitive

class TRMFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TRMFPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
