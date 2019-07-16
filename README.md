# Incremental K-Fold Train/Test Splitter for Pandas DataFrame

K-Fold Cross-Validation but with groups of incremental size

## Basic Usage:

```python
ikf = IKFold(df)
holdout = ikf.holdout(.1)
for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(ikf.split()):
      t = {}
      t['fold'] = fold_idx
      t['X_train'] = X_train
      t['y_train'] = y_train
      t['X_test']  = X_test
      t['y_test']  = y_test
      for key, value in t.items():
        if key in ['group','fold'] :
          print(key, value)
        else:
          print(key, value.shape)
      print("\tDo some processing and training here...")
```
