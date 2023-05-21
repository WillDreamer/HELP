from lib.gnn_models import GNN,Node_GCN
from lib.latent_ode import CoupledODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver,CoupledODEFunc
from lib.utils import print_parameters


def create_CoupledODE_model(args, input_dim,device):

	# dim related
	input_dim = input_dim
	output_dim = args.output_dim
	ode_hidden_dim = args.ode_dims
	rec_hidden_dim = args.rec_dims
	ode_input_dim = rec_hidden_dim

	#Encoder related
	encoder_z0 = GNN(args,in_dim=input_dim, n_hid=rec_hidden_dim)  # [b,n_ball,e]

	# ODE related
	# 1. Node ODE function
	node_ode_func_net = Node_GCN(ode_input_dim,ode_input_dim,ode_input_dim,dropout=args.dropout)

	
	# 3. Wrap Up ODE Function
	coupled_ode_func = CoupledODEFunc(
		node_ode_func_net=node_ode_func_net,
		device=device,
		num_atom = args.num_atoms,dropout=args.dropout).to(device)


	diffeq_solver = DiffeqSolver(coupled_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)
    #Decoder related
	decoder_node = Decoder(ode_input_dim, output_dim).to(device)

	model = CoupledODE(
		encoder_z0 = encoder_z0,
		ode_hidden_dim = ode_hidden_dim,
		hidden_dim = args.time,
		rec_hidden_dim = rec_hidden_dim,
		decoder_node = decoder_node,
		diffeq_solver = diffeq_solver, 
		device = device
		).to(device)

	# print_parameters(model)

	return model

