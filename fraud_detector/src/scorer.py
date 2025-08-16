import pandas as pd
import numpy as np
import os
import logging
from lightgbm import LGBMClassifier
import joblib

logger = logging.getLogger(__name__)
logger.info('Importing pretrained model...')

model = LGBMClassifier()
model = joblib.load('./models/mylightgbm_model.pkl')

model_th = 0.686
logger.info('Pretrained model imported successfully...')

def make_pred(model, dt: pd.DataFrame):
    scores = model.predict_proba(dt)[:, 1]
    preds = (scores > model_th) * 1

    return scores, preds