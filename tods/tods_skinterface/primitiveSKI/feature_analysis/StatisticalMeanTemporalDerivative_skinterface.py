import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.StatisticalMeanTemporalDerivative import StatisticalMeanTemporalDerivativePrimitive

class StatisticalMeanTemporalDerivativeSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalMeanTemporalDerivativePrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
