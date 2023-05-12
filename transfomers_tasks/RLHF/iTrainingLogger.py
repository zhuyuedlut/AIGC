import os

import numpy as np
import matplotlib.pyplot as plt

class iSummaryWriter:
    def __init__(self,
                 log_path: str,
                 log_name: str,
                 params = [],
                 extention='.png',
                 max_columns=2,
                 ):