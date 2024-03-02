from collections import defaultdict
import glob

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from metrics import get_metrics
from import_these import *


def load_data(path):
    dataset = ProteinDataset([path])
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32, drop_last=False)

    return data_loader


def get_model_files(path):
    files = glob.glob(f'./{path}/model_ckpt/checkpoint_*.ckpt')
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    assert len(sorted_files) != 0

    return sorted_files, [x.split('_')[-1].split('.')[0] for x in sorted_files]


def get_embedding(model, data_loader):
    model.eval()
    embs = []
    ys = []
    with torch.no_grad():
        for x, y in data_loader:
            hid_out = model(x,evaluation=True)[0].flatten(1).detach().cpu().tolist()
            embs.extend(hid_out)
            ys.extend(y)

    return (
        np.array(embs), # (b, c, l)
        np.array(ys)    # (b, )
    )


def save_perf(path, val_perf, test_perf, epoch_list):
    perf_dict = defaultdict(list)
    for seed in range(5):
        for e in range(len(epoch_list)):
            perf_dict['seed'].append(seed+1)
            perf_dict['epoch'].append(epoch_list[e])
            for method in ['acc', 'auc', 'mcc', 'sen', 'spec']:
                valid_epoch_perfs = [fold[method][e] for fold in val_perf[seed]]
                test_epoch_perfs = [fold[method][e] for fold in test_perf[seed]]
                perf_dict[f'{method}_test_avg'].append(round(np.mean(test_epoch_perfs), 4))
                perf_dict[f'{method}_test_std'].append(round(np.std(test_epoch_perfs), 4))
                perf_dict[f'{method}_val_avg'].append(round(np.mean(valid_epoch_perfs), 4))
                perf_dict[f'{method}_val_std'].append(round(np.std(valid_epoch_perfs), 4))
    perf_df = pd.DataFrame(perf_dict)
    perf_df.to_csv(f'./{path}/perf.csv')


if __name__ == '__main__':

    result_path = 'result'
    model_files, epoch_list = get_model_files(result_path)
    valid_performances = [[defaultdict(list) for j in range(5)] for i in range(5)]
    test_performances = [[defaultdict(list) for j in range(5)] for i in range(5)]
    device = torch.device('cuda:0')

    for seed in range(5):
        for fold in range(5):
            print(f'seed {seed} fold {fold}')

            # Load data and build a data loader
            train_loader = load_data(f'cross_val_{seed+1}/{fold+1}/cv_train_{fold+1}.csv')
            valid_loader = load_data(f'cross_val_{seed+1}/{fold+1}/cv_val_{fold+1}.csv')
            test_loader = load_data(f'independent_test/independent_test.csv')

            # Iterate over models saved at different epochs
            for files in model_files:
                print(files)

                # Get the pretrained embedding from pretrained model
                model = torch.load(files, map_location=device)
                train_embs, train_y = get_embedding(model, train_loader)
                valid_embs, valid_y = get_embedding(model, valid_loader)
                test_embs, test_y = get_embedding(model, test_loader)

                # Train a classifier and predict probabilities for validation and test data
                ml = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=30, random_state=0)
                ml.fit(train_embs, train_y)
                valid_pred_y = ml.predict_proba(valid_embs)
                test_pred_y = ml.predict_proba(test_embs)

                # Record the performance on validation and test datasets
                for m, score in get_metrics(valid_y, valid_pred_y[:, 1]).items():
                    valid_performances[seed][fold][m].append(score)
                for m, score in get_metrics(test_y, test_pred_y[:, 1]).items():
                    test_performances[seed][fold][m].append(score)

    # Save the performances to csv files
    save_perf(path, valid_performances, test_performances, epoch_list)
