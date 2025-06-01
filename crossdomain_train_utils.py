import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from .models.resnet_crossdomain import resnet18  # Import ResNet with MixStyle
from ..dataloader.crossdomain_dataset import ECGDataset, get_transforms  # Dataset with domain info
from .metrics import cal_multilabel_metrics, roc_curves
import pickle

class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initialize the device, datasets, dataloaders, model, loss, and optimizer'''
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info(f'using {self.device_count} gpu(s)')
            assert self.args.batch_size % self.device_count == 0, "Batch size should be divisible by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info(f'using {self.device_count} cpu(s)')

        # Load the datasets
        training_set = ECGDataset(self.args.train_path, get_transforms('train'))
        channels = training_set.channels
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(self.device == "cuda"),
                                   drop_last=False)

        if self.args.val_path:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'))
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=(self.device == "cuda"),
                                     drop_last=True)

        # Initialize the model with MixStyle
        self.model = resnet18(
            in_channel=channels,
            out_channel=len(self.args.labels),
            mixstyle_p=0.5,  # Probability of applying MixStyle
            mixstyle_alpha=0.1  # Beta distribution parameter for MixStyle
        )

        # Load pretrained model if specified
        if hasattr(self.args, 'load_model_path'):
            self.model.load_state_dict(torch.load(self.args.load_model_path))
            self.args.logger.info(f'Loaded the model from: {self.args.load_model_path}')
        else:
            self.args.logger.info('Training a new model from the beginning.')

        # Enable data parallelism if multiple GPUs are available
        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.model.to(self.device)

    def train(self):
        '''PyTorch training loop with MixStyle'''
        self.args.logger.info(f'train() called: model={type(self.model).__name__}, '
                              f'opt={type(self.optimizer).__name__}(lr={self.optimizer.param_groups[0]["lr"]}), '
                              f'epochs={self.args.epochs}, device={self.device}')

        history = {
            'train_loss': [],
            'train_micro_auroc': [],
            'train_micro_avg_prec': [],
            'train_macro_auroc': [],
            'train_macro_avg_prec': [],
            'train_challenge_metric': [],
            'val_loss': [],
            'val_micro_auroc': [],
            'val_micro_avg_prec': [],
            'val_macro_auroc': [],
            'val_macro_avg_prec': [],
            'val_challenge_metric': [],
        }

        start_time_sec = time.time()

        for epoch in range(1, self.args.epochs + 1):
            # Train phase
            self.model.train()
            if hasattr(self.model, 'mixstyle'):
                self.model.mixstyle.activate()

            train_loss = 0.0
            labels_all, logits_prob_all = [], []

            for ecgs, ag, labels, domains in self.train_dl:
                ecgs, ag, labels, domains = ecgs.to(self.device), ag.to(self.device), labels.to(self.device), domains.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(ecgs, ag, domains)  # Pass domain labels
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * ecgs.size(0)
                logits_prob_all.append(self.sigmoid(logits).detach())
                labels_all.append(labels.detach())

            # Aggregate results
            train_loss /= len(self.train_dl.dataset)
            logits_prob_all = torch.cat(logits_prob_all)
            labels_all = torch.cat(labels_all)

            metrics = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = metrics

            self.args.logger.info(f'Epoch {epoch}/{self.args.epochs}: Train Loss={train_loss:.4f}, '
                                  f'Train Micro AUROC={train_micro_auroc:.4f}, Train Macro AUROC={train_macro_auroc:.4f}, Train Micro Avg Prec={train_micro_avg_prec:.4f}, '
                                  f'Train Macro Avg Prec={train_macro_avg_prec:.4f}, Train Challenge Metric={train_challenge_metric:.4f}')
            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)
            history['train_challenge_metric'].append(train_challenge_metric)

            # Validation phase
            if self.args.val_path:
                self.model.eval()
                if hasattr(self.model, 'mixstyle'):
                    self.model.mixstyle.deactivate()

                val_loss = 0.0
                val_labels_all, val_logits_prob_all = [], []

                for ecgs, ag, labels, domains in self.val_dl:
                    ecgs, ag, labels = ecgs.to(self.device), ag.to(self.device), labels.to(self.device)
                    domains = domains.to(self.device)  # Ensure domain labels are also on the correct device

                    with torch.no_grad():
                        logits = self.model(ecgs, ag, domain_labels=domains)
                        loss = self.criterion(logits, labels)
                        val_loss += loss.item() * ecgs.size(0)
                        val_logits_prob_all.append(self.sigmoid(logits).detach())
                        val_labels_all.append(labels.detach())

                val_loss /= len(self.val_dl.dataset)
                val_logits_prob_all = torch.cat(val_logits_prob_all)
                val_labels_all = torch.cat(val_labels_all)

                val_metrics = cal_multilabel_metrics(val_labels_all, val_logits_prob_all, self.args.labels, self.args.threshold)
                val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc, val_challenge_metric = val_metrics

                self.args.logger.info(f'Validation Loss={val_loss:.4f}, Validation Micro AUROC={val_micro_auroc:.4f}, Validation Macro AUROC={val_macro_auroc:.4f}, '
                f'Validation Micro Avg Prec={val_micro_avg_prec:.4f}, Validation Macro Avg Prec={val_macro_avg_prec:.4f}, Validation Challenge Metric={val_challenge_metric:.4f}')
                history['val_loss'].append(val_loss)
                history['val_micro_auroc'].append(val_micro_auroc)
                history['val_macro_auroc'].append(val_macro_auroc)
                history['val_micro_avg_prec'].append(val_micro_avg_prec)
                history['val_macro_avg_prec'].append(val_macro_avg_prec)
                history['val_challenge_metric'].append(val_challenge_metric)
                
            # --------------------------------------------------------------------

            # Create ROC Curves at the beginning, middle and end of training
            if epoch == 1 or epoch == self.args.epochs/2 or epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)

            # Save a model at every 5th epoch (backup)
            if epoch in list(range(self.args.epochs)[0::5]):
                self.args.logger.info('Saved model at the epoch {}!'.format(epoch))
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '_e' + str(epoch) + '.pth')
                torch.save(model_state_dict, model_savepath)

            # Save trained model (.pth), history (.pickle) and validation logits (.csv) after the last epoch
            if epoch == self.args.epochs:
                
                self.args.logger.info('Saving the model, training history and validation logits...')
                    
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name  + '.pth')
                torch.save(model_state_dict, model_savepath)
                
                # -- Save history
                history_savepath = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_history.pickle')
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # -- Save the logits from validation if used, either save the logits from the training phase
                if self.args.val_path is not None:
                    self.args.logger.info('- Validation logits saved')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                               self.args.yaml_file_name + '_val_logits.csv') 
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_val_labels.csv') 
                    # Use filenames as indeces
                    filenames = [os.path.basename(file) for file in self.validation_files]

                else:
                    self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                               self.args.yaml_file_name + '_train_logits.csv') 
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_labels.csv') 
                    filenames = None
                
                # Save logits and corresponding labels
                labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
                labels_df.to_csv(labels_all_csv_path, sep=',')

                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
                logits_df.to_csv(logits_csv_path, sep=',')

        torch.cuda.empty_cache()
          
         
        # END OF TRAINING LOOP        
        
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))