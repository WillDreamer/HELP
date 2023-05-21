import os
import sys
from lib.load_data_ant import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from torch_geometric.data import Data, DataLoader
from lib.create_coupled_ode_model import create_CoupledODE_model
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Coupled ODE')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--datapath', type=str, default='/data1/TSP20/',\
                     choices=['/data1/TSP100/','/data/ant/data/','/data1/TSP50/'], help="default data path")
parser.add_argument('--training_end_time', type=int, default=320,
                    help="number of days between two adjacent starting date of two series.")

parser.add_argument('--num_atoms', type=int, default=20,help = 'Number of cities')
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2)
parser.add_argument('-r', '--random-seed', type=int, default=2023, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--edge_lamda', type=float, default=0.5, help='edge weight')
parser.add_argument('--time', type=int, default=10, help="Time steps.")
parser.add_argument('--output_dim', type=int, default= 100, help="Dimensionality of the output .")
parser.add_argument('--rec-dims', type=int, default= 512, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default= 100, help="Dimensionality of the ODE func for edge and node (must be the same)")
parser.add_argument('--solver', type=str, default="euler", help='dopri5,rk4,euler')

args = parser.parse_args()



############ CPU AND GPU related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:2")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

############ Feature related
args.feature_out_index = [0,1]
args.output_dim = args.num_atoms
args.ode_dims = args.num_atoms


#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # Saving Path
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    experimentID = int(SystemRandom().random() * 100000)
    ############ Command log
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)


    ################# Loading Data

    train_dataset = ParseData(args,is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = ParseData(args,is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    input_dim = args.num_atoms
    # for i in train_loader:
    #       print(i)


    ############ Model SetUp
    # Create the model
    model = create_CoupledODE_model(args, input_dim, device)
    # Load checkpoint for saved model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        print("loaded saved ckpt!")
        #exit()

    ##################################################################
    # Training

    log_path = "logs/" + "Experiment_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    currentDateAndTime = datetime.now()
    logger.info(currentDateAndTime)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    n_iters_to_viz = 1

    def train_single_batch(model,train_batch,epo):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(train_batch,args.num_atoms,epo)

        loss = train_res["loss"]
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value

    def train_epoch(epo):
        model.train()
        loss_list = []
        torch.cuda.empty_cache()

        for train_batch,target in tqdm(train_loader):

            # utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            loss = train_single_batch(model,train_batch.to(device),epo)
            #saving results
            loss_list.append(loss)
        
            del train_batch
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seqequences] | Loss {:.8f} '.format(
            epo, np.mean(loss_list))

        return message_train

    def val_epoch(epo,final=False,test=False):
        model.eval()
        torch.cuda.empty_cache()

        if final:
            ant_solution = 0
            optimal_solution = 0
            our_soultion = 0 
            for test_batch,test_target in tqdm(test_loader):
                ant_solution += test_target.min().item()
                val_res = model.compute_all_losses(test_batch.to(device), args.num_atoms,final,test)
                optimal_solution += val_res["accurate"]
                our_soultion += val_res["our_solution"]
                del test_batch
                torch.cuda.empty_cache()
            ant_solution /= len(test_loader)
            optimal_solution /= len(test_loader)
            our_soultion /= len(test_loader)
            # logger.info(val_res["accurate_tour"])
            # logger.info(val_res["our_tour"])
            message_val = 'Epoch {:04d} [Val sequence] |  Loss {:.8F} | Our Solution {:.3F}  | Ant Solution {:.3F} | Accurate Solution {:.3F}|'\
                .format(epo,val_res["loss"],our_soultion, ant_solution, optimal_solution)
        else:
            val_loss = 0
            for test_batch,test_target in tqdm(test_loader):
                
                val_res = model.compute_all_losses(test_batch.to(device), args.num_atoms,final,test)
                val_loss += val_res["loss"]
                # print(val_res["loss"],'*'*50)
                del test_batch
                torch.cuda.empty_cache()
            val_loss /= len(test_loader)
            message_val = 'Epoch {:04d} [Val sequence] |  Loss {:.8F}'.format(epo,val_loss)

        return message_val


    for epo in range(1, args.niters + 1):

        message_train = train_epoch(epo)
        message_train = train_epoch(epo)

        if (epo+1) % 10 == 0:
            message_val = val_epoch(epo,final=True,test=True)
            ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + "_epoch_" + str(epo) + "_Cities_" + str(
                    args.num_atoms) + '.ckpt')
            torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)
        else:
            message_val = val_epoch(epo,test=True)

        if epo % n_iters_to_viz == 0:
            # Logging Train and Val
            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_val)


