import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import errno
import datetime

def check_last_layer_norm(model, logger):
    last_layer_name, last_layer_param = model.classifier.named_parameters()[0]
    normal_norm = torch.norm(last_layer_param[0]).item()
    abnormal_norm = torch.norm(last_layer_param[1]).item()
    logger.info(f'Classifier Norm: ({normal_norm}, {abnormal_norm})')

class Meter:
    def __init__(self, num_epochs, save_path, plotting_save_path, test_logger):
        self.test_logger = test_logger
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.plotting_save_path = plotting_save_path
        self.test_loss_list = []
        self.test_acc_list = []
        self.test_map_list = []
        self.Best_map = 0
        self.Best_epoch = 0
        self.thr = 0.5
        self.test_acc_per_cls = []
        # self.label_names = ['NoGas', 'Perfume', 'Smoke', 'Mixture']
        self.label_names = ['정상', '관심', '경고', '위험']


    def update(self, model, preds, labels, test_loss, epoch):
        self.epoch = epoch
        self.test_loss_list.append(test_loss)
        test_targets = labels.argmax(dim=1)
        test_preds = preds.argmax(dim=1)
        correct = test_preds.eq(test_targets).sum().item()
        self.test_acc_list.append((correct / test_targets.shape[0]))
        self.test_acc_per_cls = self.compute_class_accuracy(test_targets, test_preds)

        self.test_map_list.append(self.compute_map(labels, preds))
        if self.Best_map < self.test_map_list[-1]:
            self.Best_map = self.test_map_list[-1]
            self.Best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
        
        # Log             
        self.test_logger.info(f'Epoch: {epoch}/{self.num_epochs}, Test Loss: {test_loss:.6f}, Test Accuracy: {self.test_acc_list[-1]}, Test Accuracy per class: {self.label_names} = {self.test_acc_per_cls}\n')
        
        if epoch == 99:
            self.test_logger.info(f'[Average] Average accuracy of last 10 epochs:{(sum(self.test_acc_list[-10:])/10):.4f}')

        # Plot and save
        self.plot_and_save(range(epoch+1), self.test_loss_list, 'Epoch', 'Loss', os.path.join(self.plotting_save_path, 'test_loss.png'), title='Test Loss')
        self.plot_and_save(range(epoch+1), self.test_acc_list, 'Epoch', 'Accuracy', os.path.join(self.plotting_save_path, 'test_acc.png'), title='Test Accuracy')

    def compute_class_accuracy(self, y_true, y_pred):
        class_accuracies = np.zeros(len(np.unique(y_true)))
        num_samples_per_class = np.zeros(len(np.unique(y_true)))
        for i in range(len(np.unique(y_true))):
            tmp_class_indices = np.where(y_true == i)
            class_accuracies[i] = y_pred[tmp_class_indices].eq(y_true[tmp_class_indices]).sum().item()
            num_samples_per_class[i] = y_true[tmp_class_indices].shape[0]
            class_accuracies[i] = (class_accuracies[i] / num_samples_per_class[i]) * 100
        return class_accuracies

    def compute_ap(self, gt, pred):
        precision, recall, _ = precision_recall_curve(gt, pred)
        return auc(recall, precision)

    def compute_map(self, y_true, y_pred):
        APs = []
        for i in range(y_true.shape[1]):
            APs.append(self.compute_ap(y_true[:, i], y_pred[:, i]))
        return np.mean(APs)

    def plot_and_save(self, x, y, x_label, y_label, save_path, title=None):
        plt.plot(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(save_path)
        plt.close()


def map_value(target):
    mapping = {0: '정상', 1: '관심', 2: '경고', 3: '위험'}
    return mapping[target]


def tsne_plot(save_dir, targets, outputs, comment, epoch=None):
    time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    print('generating t-SNE plot...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    
    for cls in range(len(np.unique(targets))):
        gastype = ['정상', '관심', '경고', '위험']
        targets

    vectorized_map_value = np.vectorize(map_value)
    targets_name = vectorized_map_value(targets)

    df['targets'] = targets
    df['targets_name'] = targets_name
    plt.rcParams['figure.figsize'] = 8, 8
    
    class_means = []

    for cls in range(len(np.unique(targets))):
        class_mean = tsne_output[targets == cls].mean(axis=0)
        class_means.append(class_mean)
    class_means = np.array(class_means)
    
    sns.scatterplot(
        x='x', y='y',
        hue='targets_name',
        hue_order = ['정상', '관심', '경고', '위험'],
        palette=sns.color_palette('husl', 4),
        data=df,
        marker='o',
        legend="full",
        alpha=1
    )
    colors = ['r', 'goldenrod', 'darkcyan', 'mediumslateblue']
    
    for cls in range(len(np.unique(targets))):
        gastype = ['정상', '관심', '경고', '위험']
        plt.scatter(class_means[cls, 0], class_means[cls, 1], c=colors[cls], edgecolor='k', s=450, marker='*', label = f'Center of {gastype[cls]}')

    plt.legend(fontsize = 12.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    mkdir_if_missing(save_dir)
    plt.savefig(os.path.join(save_dir,f'{time}_tsne_{comment}.png'), bbox_inches='tight')
    plt.close()
    print(f'Saved T-SNE')


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise