from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback
import numpy as np

class ROC_callback(Callback):
    
    def __init__(self, train_gen, val_gen, train_dps, val_dps, workers=1):
        
        self.train_gen = train_gen
        self.val_gen   = val_gen

        self.train_labels = np.array([dp.get_label() for dp in train_dps])
        self.val_labels = np.array([dp.get_label() for dp in val_dps])

        self.workers = workers

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def print_stats(self, labels, preds):

        print('roc auc: ', roc_auc_score(labels, preds))
        preds = np.array(preds).round()

        print('f1 score: ', f1_score(labels, preds))
        print('precision score: ', precision_score(labels, preds))
        print('recall_score: ',  recall_score(labels, preds))
        print('acc : ', accuracy_score(labels, preds))

    def on_epoch_end(self, epoch, logs={}):

        print('-- Train Metrics --')
        train_preds = self.model.predict_generator(self.train_gen, workers=self.workers)
        self.print_stats(self.train_labels, train_preds)

        print('-- Val Metrics --')
        val_preds = self.model.predict_generator(self.val_gen, workers=self.workers)
        self.print_stats(self.val_labels, val_preds)
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return