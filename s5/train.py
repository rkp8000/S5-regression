from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
import numpy
import os

from .train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, BatchRegressionModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.problem_type]

    # Dataset dependent logic
    padded = True
    
    if args.problem_type == 'rgr':
        regression = True
    elif args.problem_type == 'clf':
        regression = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, d_out, seq_len, in_dim, train_size = \
      create_dataset_fn(args.data_dir, args.cache_dir, seed=args.jax_seed, bsz=args.bsz, clear_cache=args.clear_cache)

    print(f"[*] Starting S5 {args.problem_type} training on `{args.data_dir}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm_init_fn = init_S5SSM(H=args.d_model,
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args.C_init,
                             discretization=args.discretization,
                             dt_min=args.dt_min,
                             dt_max=args.dt_max,
                             conj_sym=args.conj_sym,
                             clip_eigs=args.clip_eigs,
                             bidirectional=args.bidirectional)
    
    if args.problem_type == 'rgr':
        model_cls = partial(
            BatchRegressionModel,
            ssm=ssm_init_fn,
            d_output=d_out,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    elif args.problem_type == 'clf':
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=d_out,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # initialize training state
    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/args.bsz)
    
    if args.epoch_save_dir and not os.path.exists(args.epoch_save_dir):
        os.makedirs(args.epoch_save_dir)
    
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
            
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        
        return_train = int(args.save_training) > 0
        
        state, train_loss, step, train_pred, train_targ = train_epoch(state,
                                              skey,
                                              model_cls,
                                              args.problem_type,
                                              trainloader,
                                              seq_len,
                                              in_dim,
                                              args.batchnorm,
                                              lr_params,
                                              return_train)

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        val_loss, val_acc, val_pred, val_targ = validate(state,
                                     model_cls,
                                     args.problem_type,
                                     valloader,
                                     seq_len,
                                     in_dim,
                                     args.batchnorm)

        print(f"[*] Running Epoch {epoch + 1} Test...")
        test_loss, test_acc, test_pred, test_targ = validate(state,
                                       model_cls,
                                       args.problem_type,
                                       testloader,
                                       seq_len,
                                       in_dim,
                                       args.batchnorm)

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
            f" Val Accuracy: {val_acc:.4f} -- Test Accuracy: {test_acc:.4f}"
        )

        if args.epoch_save_dir:
            
            save_dict = {
                'val_pred': val_pred, 'val_targ': val_targ,
                'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss
            }
            
            if int(args.save_training) and ((epoch % int(args.save_training)) == 0):
                save_dict['train_pred'] = train_pred
                save_dict['train_targ'] = train_targ
                
            numpy.save(
                os.path.join(args.epoch_save_dir, f'epoch_{epoch}.npy'), numpy.array([save_dict]))
            
        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_val_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if count > args.early_stop_patience:
            break
