import numpy as np
import os, sys
import time
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from .models.resnet_crossdomain import resnet18  
from ..dataloader.crossdomain_dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle

class Predicting(object):
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        ''' Initialize the device conditions and dataloader, loading trained model '''
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info(f'using {self.device_count} gpu(s)')
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using 1 cpu')

        filenames = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
        self.filenames = [f for file in filenames for f in file]

        #load the test data
        testing_set = ECGDataset(self.args.test_path, get_transforms('test'))
        channels = testing_set.channels
        self.test_dl = DataLoader(
            testing_set,
            batch_size=1,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
            drop_last=False
        )

        #load the trained model
        self.model = resnet18(in_channel=channels, out_channel=len(self.args.labels))
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))

        #set the model to evaluation mode and deactivate MixStyle if present
        self.model.eval()
        if hasattr(self.model, 'mixstyle'):
            self.model.mixstyle.deactivate()

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)

    def predict(self):
        ''' Make predictions '''
        self.args.logger.info(f'predict() called: model={type(self.model).__name__}, device={self.device}')

        #initialize history
        history = {
            'test_micro_avg_prec': 0.0,
            'test_micro_auroc': 0.0,
            'test_macro_avg_prec': 0.0,
            'test_macro_auroc': 0.0,
            'test_challenge_metric': 0.0,
            'labels': self.args.labels,
            'test_csv': self.args.test_path,
            'threshold': self.args.threshold
        }

        start_time_sec = time.time()

        # --- evaluate on Testing Set ---
        self.model.eval()  # Ensure MixStyle remains deactivated
        labels_all = torch.tensor([], device=self.device)
        logits_prob_all = torch.tensor([], device=self.device)

        for i, (ecgs, ag, labels, domains) in enumerate(self.test_dl):  #include domains in the dataset
            ecgs = ecgs.to(self.device)  #ECG data
            ag = ag.to(self.device)      #age and gender metadata
            labels = labels.to(self.device)  #diagnoses in SNOMED CT codes 
            domains.to(self.device)

            with torch.no_grad():
                logits = self.model(ecgs, ag, domains)
                logits_prob = self.sigmoid(logits)
                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            if i % 1000 == 0:
                self.args.logger.info(f'{i + 1}/{len(self.test_dl)} predictions made')

        # alculate metrics
        test_metrics = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc, test_challenge_metric = test_metrics

        self.args.logger.info(
            f'macro avg prec: {test_macro_avg_prec:.2f}, micro avg prec: {test_micro_avg_prec:.2f}, '
            f'macro auroc: {test_macro_auroc:.2f}, micro auroc: {test_micro_auroc:.2f}, '
            f'challenge metric: {test_challenge_metric:.2f}'
        )

        #draw ROC curve for predictions
        roc_curves(labels_all, logits_prob_all, self.args.labels, epoch=None, save_path=self.args.roc_save_dir)

        #update testing history with metrics
        history.update({
            'test_micro_auroc': test_micro_auroc,
            'test_micro_avg_prec': test_micro_avg_prec,
            'test_macro_auroc': test_macro_auroc,
            'test_macro_avg_prec': test_macro_avg_prec,
            'test_challenge_metric': test_challenge_metric
        })

        #save the history
        history_savepath = os.path.join(self.args.output_dir, f"{self.args.yaml_file_name}_test_history.pickle")
        with open(history_savepath, mode='wb') as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

        #save labels and logits
        filenames = [os.path.basename(file) for file in self.filenames]
        logits_csv_path = os.path.join(self.args.output_dir, f"{self.args.yaml_file_name}_test_logits.csv")
        labels_csv_path = os.path.join(self.args.output_dir, f"{self.args.yaml_file_name}_test_labels.csv")

        logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
        logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
        logits_df.to_csv(logits_csv_path, sep=',')

        labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
        labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
        labels_df.to_csv(labels_csv_path, sep=',')

        torch.cuda.empty_cache()

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        self.args.logger.info(f'Total time: {total_time_sec:.2f} seconds')
