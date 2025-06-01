import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from .models.resnet_dann import seresnet18_dann, ResNetDANN, BasicBlock  # updated import
from ..dataloader.dann_dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle

class Training(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders,
        model, loss, criterion and optimizer
        '''

        # consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))

        # load the datasets
        training_set = ECGDataset(self.args.train_path, get_transforms('train'))
        channels = training_set.channels
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=False)

        if self.args.val_path is not None:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'))
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=(True if self.device == 'cuda' else False),
                                     drop_last=True)

        # update model initialization to ResNetDANN
        self.model = ResNetDANN(BasicBlock, [2, 2, 2, 2], in_channel=channels, out_channel=len(self.args.labels))

        # load model if necessary
        if hasattr(self.args, 'load_model_path'):
            self.model.load_state_dict(torch.load(self.args.load_model_path))
            self.args.logger.info('Loaded the model from: {}'.format(self.args.load_model_path))
        else:
            self.args.logger.info('Training a new model from the beginning.')

        # if more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)

        self.criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()  # added domain criterion
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)

    def train(self):
        ''' PyTorch training loop
        '''

        self.args.logger.info('train() called: model={}, opt={}(lr={}), epochs={}, device={}'.format(
            type(self.model).__name__,
            type(self.optimizer).__name__,
            self.optimizer.param_groups[0]['lr'],
            self.args.epochs,
            self.device))

        # add all wanted history information
        history = {}
        history['train_csv'] = self.args.train_path
        history['train_loss'] = []
        history['train_class_loss'] = []  # Added for classification loss
        history['train_domain_loss'] = []  # Added for domain loss
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = []
        history['train_challenge_metric'] = []

        if self.args.val_path is not None:
            history['val_csv'] = self.args.val_path
            history['val_loss'] = []
            history['val_class_loss'] = []  # Added for classification loss
            history['val_domain_loss'] = []  # Added for domain loss
            history['val_micro_auroc'] = []
            history['val_micro_avg_prec'] = []
            history['val_macro_auroc'] = []
            history['val_macro_avg_prec'] = []
            history['val_challenge_metric'] = []

        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer'] = self.optimizer
        history['criterion'] = self.criterion

        start_time_sec = time.time()

        for epoch in range(1, self.args.epochs+1):
            # --- TRAIN ON TRAINING SET ------------------------------------------
            self.model.train()
            train_loss = 0.0
            train_class_loss = 0.0  # Added for classification loss
            train_domain_loss = 0.0  # Added for domain loss
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)

            batch_loss = 0.0
            batch_count = 0
            step = 0

            for batch_idx, (ecgs, ag, labels, domains) in enumerate(self.train_dl):
                ecgs = ecgs.to(self.device)  # ECGs
                ag = ag.to(self.device)  # age and gender
                labels = labels.to(self.device)  # diagnoses in SNOMED CT codes
                domains = domains.to(self.device)  # domains

                with torch.set_grad_enabled(True):
                    class_output, domain_output = self.model(ecgs, ag, alpha=0.3)  # adjust alpha as needed
                    loss_class = self.criterion(class_output, labels)
                    loss_domain = self.domain_criterion(domain_output, domains)
                    loss = loss_class + 5 * loss_domain
                    logits_prob = self.sigmoid(class_output)
                    loss_tmp = loss.item() * ecgs.size(0)
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                    train_loss += loss_tmp
                    train_class_loss += loss_class.item() * ecgs.size(0)  # Added for classification loss
                    train_domain_loss += loss_domain.item() * ecgs.size(0)  # Added for domain loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # printing training information
                    if step % 100 == 0:
                        batch_loss += loss_tmp
                        batch_count += ecgs.size(0)
                        batch_loss = batch_loss / batch_count
                        self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                            epoch,
                            batch_idx * len(ecgs),
                            len(self.train_dl.dataset),
                            batch_loss
                        ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            train_loss = train_loss / len(self.train_dl.dataset)
            train_class_loss = train_class_loss / len(self.train_dl.dataset)  # Added for classification loss
            train_domain_loss = train_domain_loss / len(self.train_dl.dataset)  # Added for domain loss
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train class loss: {:<6.2f}  train domain loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f} train micro avg prec: {:<6.2f} train macro auroc: {:<6.2f} train macro avg prec: {:<6.2f}'.format(
                epoch,
                self.args.epochs,
                train_loss,
                train_class_loss,
                train_domain_loss,
                train_micro_auroc,
                train_challenge_metric,
                train_micro_avg_prec,
                train_macro_auroc,
                train_macro_avg_prec))

            # add information for training history
            history['train_loss'].append(train_loss)
            history['train_class_loss'].append(train_class_loss)  # Added for classification loss
            history['train_domain_loss'].append(train_domain_loss)  # Added for domain loss
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)
            history['train_challenge_metric'].append(train_challenge_metric)

            # --- EVALUATE ON VALIDATION SET -------------------------------------
            if self.args.val_path is not None:
                self.model.eval()
                val_loss = 0.0
                val_class_loss = 0.0  # Added for classification loss
                val_domain_loss = 0.0  # Added for domain loss
                labels_all = torch.tensor((), device=self.device)
                logits_prob_all = torch.tensor((), device=self.device)

                for ecgs, ag, labels, domains in self.val_dl:
                    ecgs = ecgs.to(self.device)  # ECGs
                    ag = ag.to(self.device)  # age and gender
                    labels = labels.to(self.device)  # diagnoses in SNOMED CT codes
                    domains = domains.to(self.device)  # domains

                    with torch.set_grad_enabled(False):
                        class_output, domain_output = self.model(ecgs, ag, alpha=0.3)  # adjust alpha as needed
                        loss_class = self.criterion(class_output, labels)
                        loss_domain = self.domain_criterion(domain_output, domains)
                        loss = loss_class + 5 * loss_domain
                        logits_prob = self.sigmoid(class_output)
                        val_loss += loss.item() * ecgs.size(0)
                        val_class_loss += loss_class.item() * ecgs.size(0)  # Added for classification loss
                        val_domain_loss += loss_domain.item() * ecgs.size(0)  # Added for domain loss
                        labels_all = torch.cat((labels_all, labels), 0)
                        logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                val_loss = val_loss / len(self.val_dl.dataset)
                val_class_loss = val_class_loss / len(self.val_dl.dataset)  # Added for classification loss
                val_domain_loss = val_domain_loss / len(self.val_dl.dataset)  # Added for domain loss
                val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc, val_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

                self.args.logger.info('                val loss:  {:<6.2f}   val class loss: {:<6.2f}   val domain loss: {:<6.2f}   val micro auroc: {:<6.2f}    val challenge metric:  {:<6.2f} val micro avg prec: {:<6.2f} val macro auroc: {:<6.2f} val macro avg prec: {:<6.2f}'.format(
                    val_loss,
                    val_class_loss,
                    val_domain_loss,
                    val_micro_auroc,
                    val_challenge_metric,
                    val_micro_avg_prec,
                    val_macro_auroc,
                    val_macro_avg_prec))

                history['val_loss'].append(val_loss)
                history['val_class_loss'].append(val_class_loss)  # Added for classification loss
                history['val_domain_loss'].append(val_domain_loss)  # Added for domain loss
                history['val_micro_auroc'].append(val_micro_auroc)
                history['val_micro_avg_prec'].append(val_micro_avg_prec)
                history['val_macro_auroc'].append(val_macro_auroc)
                history['val_macro_avg_prec'].append(val_macro_avg_prec)
                history['val_challenge_metric'].append(val_challenge_metric)

            # --------------------------------------------------------------------

            # create ROC Curves at the beginning, middle, and end of training
            if epoch == 1 or epoch == self.args.epochs / 2 or epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)

            # save a model at every 5th epoch (backup)
            if epoch in list(range(self.args.epochs)[0::5]):
                self.args.logger.info('Saved model at the epoch {}!'.format(epoch))
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '_e' + str(epoch) + '.pth')
                torch.save(model_state_dict, model_savepath)

            # save trained model (.pth), history (.pickle) and validation logits (.csv) after the last epoch
            if epoch == self.args.epochs:
                self.args.logger.info('Saving the model, training history and validation logits...')

                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '.pth')
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
                    filenames = [os.path.basename(file) for file in self.validation_files]

                else:
                    self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                                   self.args.yaml_file_name + '_train_logits.csv')
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                       self.args.yaml_file_name + '_train_labels.csv')
                    filenames = None

                # save logits and corresponding labels
                labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
                labels_df.to_csv(labels_all_csv_path, sep=',')

                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
                logits_df.to_csv(logits_csv_path, sep=',')

        torch.cuda.empty_cache()

        # END OF TRAINING LOOP

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total: %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))