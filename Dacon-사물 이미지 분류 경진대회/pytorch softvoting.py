import numpy as np
import pandas as pd


sample_submission = pd.read_csv('data/sample_submission.csv')

results1 = np.load("resnet50.npy")
results2 = np.load("eff0.npy")
results3 = np.load("eff3.npy")
results4 = np.load("eff4.npy")

results = results1 + results2
max_indices = np.argmax(results, axis=1)

sample_submission.iloc[:, 1:] = max_indices[:,np.newaxis]



labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
          5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
sample_submission['target'] = sample_submission['target'].map(labels)
sample_submission.to_csv('baseline_pytorch.csv', index=False)