import argparse
from s5.utils.util import str2bool
from s5.train import train
from s5.dataloading import Datasets

        
class TrainArgs(object):
    
    def __init__(self, **kwargs):
        
        self.cache_dir = kwargs.get('cache_dir', './cache_dir')
        self.data_dir = kwargs.get('data_dir', None)
        self.clear_cache = kwargs.get('clear_cache', None)
        self.epoch_save_dir = kwargs.get('epoch_save_dir', '')
        self.save_training = kwargs.get('save_training', 0)
        
        self.problem_type = kwargs.get('problem_type', None)
        self.n_layers = kwargs.get('n_layers', 6)
        self.d_model = kwargs.get('d_model', 128)
        self.ssm_size_base = kwargs.get('ssm_size_base', 256)
        self.blocks = kwargs.get('blocks', 8)
        self.C_init = kwargs.get('C_init', 'trunc_standard_normal')
        self.discretization = kwargs.get('discretization', 'zoh')
        self.mode = kwargs.get('mode', 'pool')
        self.activation_fn = kwargs.get('activation_fn', 'half_glu1')
        self.conj_sym = kwargs.get('conj_sym', True)
        self.clip_eigs = kwargs.get('clip_eigs', False)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.dt_min = kwargs.get('dt_min', .001)
        self.dt_max = kwargs.get('dt_max', 0.1)
        
        self.prenorm = kwargs.get('prenorm', True)
        self.batchnorm = kwargs.get('batchnorm', True)
        self.bn_momentum = kwargs.get('bn_momentum', .95)
        self.bsz = kwargs.get('bsz', 64)
        self.epochs = kwargs.get('epochs', 100)
        self.early_stop_patience = kwargs.get('early_stop_patience', 1000)
        self.ssm_lr_base = kwargs.get('ssm_lr_base', 1e-3)
        self.lr_factor = kwargs.get('lr_factor', 1)
        self.dt_global = kwargs.get('dt_global', False)
        self.lr_min = kwargs.get('lr_min', 0)
        self.cosine_anneal = kwargs.get('cosine_anneal', True)
        self.warmup_end = kwargs.get('warmup_end', 1)
        self.lr_patience = kwargs.get('lr_patience', 1000000)
        self.reduce_factor = kwargs.get('reduce_factor', 1.0)
        self.p_dropout = kwargs.get('p_dropout', 0.0)
        self.weight_decay = kwargs.get('weight_decay', 0.05)
        self.opt_config = kwargs.get('opt_config', 'standard')
        
        self.jax_seed = kwargs.get('jax_seed', 5464358)
        

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--cache_dir", type=str, default='./cache_dir',
						help="name of directory where data is cached")
	parser.add_argument("--data_dir", type=str, default=None,
						help="dataset directory")
	parser.add_argument("--clear_cache", type=str, default=False,
						help="clear the cache before executing?")
	parser.add_argument("--epoch_save_dir", type=str, default='',
						help="where to save epoch data")
	parser.add_argument("--save_training", type=str, default=0,
						help="how often to save training predictions")

	# Model Parameters
	parser.add_argument("--problem_type", type=str, default=None,
						help="Classication (clf) or Regression (rgr)")
	parser.add_argument("--n_layers", type=int, default=6,
						help="Number of layers in the network")
	parser.add_argument("--d_model", type=int, default=128,
						help="Number of features, i.e. H, "
							 "dimension of layer inputs/outputs")
	parser.add_argument("--ssm_size_base", type=int, default=256,
						help="SSM Latent size, i.e. P")
	parser.add_argument("--blocks", type=int, default=8,
						help="How many blocks, J, to initialize with")
	parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
						choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
						help="Options for initialization of C: \\"
							 "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
							 "lecun_normal sample from lecun normal, then multiply by V\\ " \
							 "complex_normal: sample directly from complex standard normal")
	parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
	parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"],
						help="options: (for classification tasks) \\" \
							 " pool: mean pooling \\" \
							 "last: take last element")
	parser.add_argument("--activation_fn", default="half_glu1", type=str,
						choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
	parser.add_argument("--conj_sym", type=str2bool, default=True,
						help="whether to enforce conjugate symmetry")
	parser.add_argument("--clip_eigs", type=str2bool, default=False,
						help="whether to enforce the left-half plane condition")
	parser.add_argument("--bidirectional", type=str2bool, default=False,
						help="whether to use bidirectional model")
	parser.add_argument("--dt_min", type=float, default=0.001,
						help="min value to sample initial timescale params from")
	parser.add_argument("--dt_max", type=float, default=0.1,
						help="max value to sample initial timescale params from")

	# Optimization Parameters
	parser.add_argument("--prenorm", type=str2bool, default=True,
						help="True: use prenorm, False: use postnorm")
	parser.add_argument("--batchnorm", type=str2bool, default=True,
						help="True: use batchnorm, False: use layernorm")
	parser.add_argument("--bn_momentum", type=float, default=0.95,
						help="batchnorm momentum")
	parser.add_argument("--bsz", type=int, default=64,
						help="batch size")
	parser.add_argument("--epochs", type=int, default=100,
						help="max number of epochs")
	parser.add_argument("--early_stop_patience", type=int, default=1000,
						help="number of epochs to continue training when val loss plateaus")
	parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
						help="initial ssm learning rate")
	parser.add_argument("--lr_factor", type=float, default=1,
						help="global learning rate = lr_factor*ssm_lr_base")
	parser.add_argument("--dt_global", type=str2bool, default=False,
						help="Treat timescale parameter as global parameter or SSM parameter")
	parser.add_argument("--lr_min", type=float, default=0,
						help="minimum learning rate")
	parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
	parser.add_argument("--warmup_end", type=int, default=1,
						help="epoch to end linear warmup")
	parser.add_argument("--lr_patience", type=int, default=1000000,
						help="patience before decaying learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--reduce_factor", type=float, default=1.0,
						help="factor to decay learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--p_dropout", type=float, default=0.0,
						help="probability of dropout")
	parser.add_argument("--weight_decay", type=float, default=0.05,
						help="weight decay value")
	parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
																			   'BandCdecay',
																			   'BfastandCdecay',
																			   'noBCdecay'],
						help="Opt configurations: \\ " \
			   "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
	  	       "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
	parser.add_argument("--jax_seed", type=int, default=1919,
						help="seed randomness")

	train(parser.parse_args())