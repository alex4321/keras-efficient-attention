import os
if os.environ.get('TF_KERAS'):
    import tensorflow.keras as keras
else:
    import keras
