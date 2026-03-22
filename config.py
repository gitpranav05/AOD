# from utils import str2bool
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--ori_data_path', type=str, default='RESIDE-6K/train/clear',  help='Origin image path')
# parser.add_argument('--haze_data_path', type=str, default='RESIDE-6K/train/hazy',  help='Haze image path')
# parser.add_argument('--val_ori_data_path', type=str, default='RESIDE-6K/val/clear',  help='Validation origin image path')
# parser.add_argument('--val_haze_data_path', type=str, default='RESIDE-6K/val/hazy',  help='Validation haze image path')
# parser.add_argument('--sample_output_folder', type=str, default='samples',  help='Validation haze image path')
# parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
# parser.add_argument('--gpu', type=int, default=0, help='GPU id')
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
# parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for data loader, for window set to 0')
# parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss ')
# parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
# parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
# parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
# parser.add_argument('--model_dir', type=str, default='./model')
# parser.add_argument('--log_dir', type=str, default='./log')
# parser.add_argument('--ckpt', type=str, default='./model/nets/net_1.pkl')
# parser.add_argument('--net_name', type=str, default='nets')
# parser.add_argument('--weight_decay', type=float, default=0.0001)
# parser.add_argument('--grad_clip_norm', type=float, default=0.1)


# def get_config():
#     config, unparsed = parser.parse_known_args()
#     return config, unparsed




from utils import str2bool
import argparse

parser = argparse.ArgumentParser()

# ===================== DATA PATHS =====================
parser.add_argument(
    '--ori_data_path',
    type=str,
    default='RESIDE-6K/train/clear',
    help='Training clear image path'
)

parser.add_argument(
    '--haze_data_path',
    type=str,
    default='RESIDE-6K/train/hazy',
    help='Training hazy image path'
)

parser.add_argument(
    '--val_ori_data_path',
    type=str,
    default='RESIDE-6K/val/clear',
    help='Validation clear image path'
)

parser.add_argument(
    '--val_haze_data_path',
    type=str,
    default='RESIDE-6K/val/hazy',
    help='Validation hazy image path'
)

parser.add_argument(
    '--sample_output_folder',
    type=str,
    default='samples',
    help='Folder to save validation samples'
)

# ===================== GPU SETTINGS =====================
parser.add_argument(
    '--use_gpu',
    type=str2bool,
    default=True,
    help='Use GPU if CUDA is available'
)

parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='GPU id (default: 0)'
)

# ===================== TRAINING SETTINGS =====================
parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
    help='Learning rate (default: 1e-4)'
)

# 🚨 IMPORTANT FOR WINDOWS 🚨
# num_workers > 0 causes shared-memory crashes on Windows
parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='Number of DataLoader workers (Windows-safe: MUST be 0)'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=8,
    help='Training batch size (safe for GPU + Windows)'
)

parser.add_argument(
    '--val_batch_size',
    type=int,
    default=8,
    help='Validation batch size'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of training epochs'
)

parser.add_argument(
    '--print_gap',
    type=int,
    default=50,
    help='Number of steps between loss prints'
)

# ===================== MODEL / OPTIM =====================
parser.add_argument(
    '--model_dir',
    type=str,
    default='./model',
    help='Directory to save models'
)

parser.add_argument(
    '--log_dir',
    type=str,
    default='./log',
    help='Directory to save logs'
)

parser.add_argument(
    '--ckpt',
    type=str,
    default='./model/nets/net_1.pkl',
    help='Checkpoint path'
)

parser.add_argument(
    '--net_name',
    type=str,
    default='nets',
    help='Network name'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='Weight decay'
)

parser.add_argument(
    '--grad_clip_norm',
    type=float,
    default=0.1,
    help='Gradient clipping norm'
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
