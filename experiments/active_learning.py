import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import time
import argparse

parser = argparse.ArgumentParser(description='SAIA experiment for the HH-VAEM model')

parser.add_argument('--model', type=str, default='VAEM',
                    help='model to use (VAE, HVAE, HMCVAE, HHVAE, VAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--dataset', type=str, default='boston', 
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--split', type=int, default=0,
                    help='train/test split index to use (default splits: 0, ..., 9)')
parser.add_argument('--version', type=str, default='version_0', 
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--method', type=str, default="mi",
                    help='method to use for Information Reward (mi or kl)')  
parser.add_argument('--set', type=str, default='test',
                    help='train or test set')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

args.cuda = args.gpu and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')

config = configs_active_learning[args.dataset]
samples = config['samples']
bins = config['bins']
step = config['step']

start = time.time()
if __name__ == '__main__':

    with torch.no_grad():

        results = []
        ckpt_path = find_path(args.dataset, args.model, args.split, args.version)
        model = load_model(args.model, ckpt_path, device).eval()
        
        mixed = None
        if args.model.__contains__('VAEM'):
            mixed = True

        dataloader = get_dataset_loader(model.dataset, args.set, path=model.data_path, batch_size=model.batch_size, split_idx=model.split_idx, dim=None, mixed=mixed)
        tqdm_batch = tqdm(total=len(dataloader), desc='Batch', position=0, leave=False)
        metric_batches = []
        ll_batches = []
        rand_metric_batches = []
        rand_ll_batches = []
        times_batches = []

        for batch in dataloader:
            metric = []
            ll = []
            rand_metric = []
            rand_ll = []
            # AL using the MI method
            if args.method == 'mi':
                metric, ll, times = model.active_learning(batch, samples=samples, bins=bins, step=step)
            # AL using the KL method
            else:
                metric, ll, times= active_learning_kl(model, batch, samples=samples, step=step)
            # Randomly adding features
            rand_metric, rand_ll = random_learning(batch, model, K_metric=samples, step=step)

            metric_batches.append(metric)
            ll_batches.append(ll)
            rand_metric_batches.append(rand_metric)
            rand_ll_batches.append(rand_ll)
            times_batches.append(times)
            tqdm_batch.update()
        
        metric = torch.stack(metric_batches)     # batches x dims
        metric = metric.mean(dim=0)

        ll = torch.stack(ll_batches, dim=0)       # batches x dims
        ll = ll.mean(dim=0)                     

        rand_metric = torch.stack(rand_metric_batches)
        rand_metric = rand_metric.mean(dim=0)

        rand_ll = torch.stack(rand_ll_batches, dim=0)
        rand_ll = rand_ll.mean(dim=0)

        times = np.stack(times_batches)
        times = times.mean(axis=0)

        results = {
            'metric': metric.detach().cpu().numpy(),
            'll': ll.detach().cpu().numpy(),
            'rand_metric': rand_metric.detach().cpu().numpy(),
            'rand_ll': rand_ll.detach().cpu().numpy(),
            'times': times
        }

        name = 'active_learning_{}_{}'.format(args.method, args.set)
        np.save(ckpt_path.split('checkpoints', 1)[0] + name, results)