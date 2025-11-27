import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from configs import Config
from data import Data_Centerline_Graph, CarotidArtery_Centerline
from nn import DGCNN_CenterGraph, CarotidArtery_CenterGraph
from utils.G3e_Net_utils import write_log, get_epoch_str


def ddp_setup():
    """Initialize the distributed environment for DDP training."""
    os.environ["MASTER_ADDR"] = "localhost"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def create_model(config, model_type='dgcnn', device=None):
    """Build the requested model and wrap it with SyncBatchNorm + DDP."""
    if device is None:
        device = int(os.environ['LOCAL_RANK'])
    
    model_params = {
        'k': config.net['k'],
        'dropout': config.net['dropout'],
        'output_channels': config.net['output_channels'],
        'n_features': config.net['n_features'],
        'batch_size': config.net['batch_size'],
        'hyper_parameters': config.net['hyper_parameters'],
        'emb_dims': config.net['emb_dims']
    }
    
    if model_type == 'carotid':
        model = CarotidArtery_CenterGraph(**model_params).to(device)
    else:
        model = DGCNN_CenterGraph(**model_params).to(device)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    if device == 0:
        total_params = sum(p.data.nelement() for p in model.parameters())
        print(f'Parameter count: {total_params}')
    
    return model


def create_dataloaders(config, dataset_class, device=None):
    """Create distributed training/validation dataloaders."""
    if device is None:
        device = int(os.environ['LOCAL_RANK'])
    
    # Training splits
    train_observation = dataset_class(config, 'train', 'observation', 20000, 7, "all")
    train_boundary = dataset_class(config, 'train', 'boundary', 10000, 7, "all")
    
    obs_sampler = DistributedSampler(train_observation)
    bou_sampler = DistributedSampler(train_boundary)
    
    train_loader_ob = DataLoader(
        dataset=train_observation,
        batch_size=config.training['batchsize'],
        shuffle=False,
        pin_memory=True,
        num_workers=config.training['n_cpu'],
        drop_last=True,
        sampler=obs_sampler
    )
    train_loader_bo = DataLoader(
        dataset=train_boundary,
        batch_size=config.training['batchsize'],
        shuffle=False,
        pin_memory=True,
        num_workers=config.training['n_cpu'],
        drop_last=True,
        sampler=bou_sampler
    )
    
    # Validation splits
    valid_dataset_ob = dataset_class(config, 'valid', 'observation', 20000, 7, "all")
    valid_dataset_bo = dataset_class(config, 'valid', 'boundary', 10000, 7, "all")
    
    valid_sampler_ob = DistributedSampler(valid_dataset_ob)
    valid_sampler_bo = DistributedSampler(valid_dataset_bo)
    
    valid_loader_ob = DataLoader(
        dataset=valid_dataset_ob,
        batch_size=config.validing['batchsize'],
        shuffle=False,
        pin_memory=True,
        num_workers=16,
        sampler=valid_sampler_ob
    )
    valid_loader_bo = DataLoader(
        dataset=valid_dataset_bo,
        batch_size=config.validing['batchsize'],
        shuffle=False,
        pin_memory=True,
        num_workers=16,
        sampler=valid_sampler_bo
    )
    
    return train_loader_ob, train_loader_bo, valid_loader_ob, valid_loader_bo


def train_epoch(model, train_loader_ob, train_loader_bo, optimizer, config, device, epoch):
    """Execute a single training epoch."""
    model.train()
    epoch_ob_u = epoch_ob_v = epoch_ob_w = epoch_ob_p = epoch_loss = epoch_mae = 0.0
    
    train_dataloader = zip(train_loader_bo, train_loader_ob)
    len_loader = len(train_loader_ob)
    
    for i_batch, ((inputs_bo, targets_bo, hiddens_bo, centerlines_bo),
                  (inputs_ob, targets_ob, hiddens_ob, centerlines_ob)) in enumerate(train_dataloader):
        
        # Prepare inputs
        input_bo = [ii.clone().detach().requires_grad_(True).float().to(device) for ii in inputs_bo]
        input_ob = [ii.clone().detach().requires_grad_(True).float().to(device) for ii in inputs_ob]
        
        # Prepare targets
        target_ob = [tt.to(device) for tt in targets_ob]
        tar_u_ob, tar_v_ob, tar_w_ob = target_ob
        tar_p_ob = hiddens_ob.to(device)
        
        target_bo = [tt.to(device) for tt in targets_bo]
        centerline_bo = [cc.to(device) for cc in centerlines_bo]
        centerline_ob = [cc.to(device) for cc in centerlines_ob]
        
        # Forward pass
        u, v, w, p = model.module.forward(input_bo, input_ob, centerline_bo, centerline_ob)
        
        # Loss
        loss_u_ob = torch.mean(torch.square(u - tar_u_ob))
        loss_v_ob = torch.mean(torch.square(v - tar_v_ob))
        loss_w_ob = torch.mean(torch.square(w - tar_w_ob))
        loss_p_ob = torch.mean(torch.square(p - tar_p_ob))
        loss = loss_u_ob + loss_v_ob + loss_w_ob + loss_p_ob
        
        # MAE
        mae_u = torch.mean(torch.abs(u - tar_u_ob))
        mae_v = torch.mean(torch.abs(v - tar_v_ob))
        mae_w = torch.mean(torch.abs(w - tar_w_ob))
        mae_p = torch.mean(torch.abs(p - tar_p_ob))
        mae = mae_u + mae_v + mae_w + mae_p
        
        # Aggregate running metrics
        epoch_ob_u += loss_u_ob.item() / len_loader
        epoch_ob_v += loss_v_ob.item() / len_loader
        epoch_ob_w += loss_w_ob.item() / len_loader
        epoch_ob_p += loss_p_ob.item() / len_loader
        epoch_loss += loss.item() / len_loader
        epoch_mae += mae.item() / len_loader
        
        # Backward
        loss.backward()
        if (i_batch + 1) % config.training['grad_ac'] == 0 or (i_batch + 1) == len_loader:
            optimizer.step()
            optimizer.zero_grad()
        
        # Progress bar (rank 0 only)
        if device == 0 and ((i_batch + 1) % 1 == 0 or (i_batch + 1) == len_loader):
            print(f"Epoch {epoch}|{config.training['epochs']}, Batch {i_batch + 1}|{len_loader}: "
                  f"Loss -- {loss.item():.6f}, U -- {loss_u_ob.item():.6f}, "
                  f"V -- {loss_v_ob.item():.6f}, W -- {loss_w_ob.item():.6f}, "
                  f"P -- {loss_p_ob.item():.6f}, MAE -- {mae.item():.6f}")
    
    return epoch_loss, epoch_ob_u, epoch_ob_v, epoch_ob_w, epoch_ob_p, epoch_mae


def validate_epoch(model, valid_loader_ob, valid_loader_bo, config, device, epoch):
    """Evaluate the model on validation splits."""
    model.eval()
    epoch_loss_u = epoch_loss_v = epoch_loss_w = epoch_loss_p = 0.0
    epoch_test_loss = epoch_test_mae = epoch_test_nmae = 0.0
    
    valid_dataloader = zip(valid_loader_bo, valid_loader_ob)
    len_loader_eval = len(valid_loader_ob)
    
    with torch.no_grad():
        for i_batch, ((inputs_bo_eval, targets_bo_eval, hiddens_bo_eval, centerlines_bo),
                      (inputs_ob_eval, targets_ob_eval, hiddens_ob_eval, centerlines_ob)) in enumerate(valid_dataloader):
            
            # Prepare inputs
            input_bo_eval = [ii.clone().detach().requires_grad_(True).float().to(device) for ii in inputs_bo_eval]
            input_ob_eval = [ii.clone().detach().requires_grad_(True).float().to(device) for ii in inputs_ob_eval]
            target_ob_eval = [tt.to(device) for tt in targets_ob_eval]
            tar_u_eval, tar_v_eval, tar_w_eval = target_ob_eval
            tar_p_eval = hiddens_ob_eval.to(device)
            
            centerline_bo = [cc.to(device) for cc in centerlines_bo]
            centerline_ob = [cc.to(device) for cc in centerlines_ob]
            
            # Forward pass
            predu, predv, predw, predp = model.module.forward(input_bo_eval, input_ob_eval, centerline_bo, centerline_ob)
            
            # Loss
            loss_u = torch.mean(torch.square(predu - tar_u_eval))
            loss_v = torch.mean(torch.square(predv - tar_v_eval))
            loss_w = torch.mean(torch.square(predw - tar_w_eval))
            loss_p = torch.mean(torch.square(predp - tar_p_eval))
            loss_eval = loss_u + loss_v + loss_w + loss_p
            
            # Normalized MAE
            nmae_u = torch.mean(torch.abs(predu - tar_u_eval) / (torch.abs(predu).max() - torch.abs(predu).min() + 1e-8))
            nmae_v = torch.mean(torch.abs(predv - tar_v_eval) / (torch.abs(predv).max() - torch.abs(predv).min() + 1e-8))
            nmae_w = torch.mean(torch.abs(predw - tar_w_eval) / (torch.abs(predw).max() - torch.abs(predw).min() + 1e-8))
            nmae_p = torch.mean(torch.abs(predp - tar_p_eval) / (torch.abs(predp).max() - torch.abs(predp).min() + 1e-8))
            nmae = nmae_u + nmae_v + nmae_w + nmae_p
            
            # MAE (absolute)
            mae_predu = torch.mean(torch.abs(predu - tar_u_eval))
            mae_predv = torch.mean(torch.abs(predv - tar_v_eval))
            mae_predw = torch.mean(torch.abs(predw - tar_w_eval))
            mae_predp = torch.mean(torch.abs(predp - tar_p_eval))
            mae_pred = mae_predu + mae_predv + mae_predw + mae_predp
            
            # Aggregate metrics
            epoch_loss_u += loss_u.item() / len_loader_eval
            epoch_loss_v += loss_v.item() / len_loader_eval
            epoch_loss_w += loss_w.item() / len_loader_eval
            epoch_loss_p += loss_p.item() / len_loader_eval
            epoch_test_loss += loss_eval.item() / len_loader_eval
            epoch_test_nmae += nmae.item() / len_loader_eval
            epoch_test_mae += mae_pred.item() / len_loader_eval
            
            # Progress bar (rank 0 only)
            if device == 0 and ((i_batch + 1) % 1 == 0 or (i_batch + 1) == len_loader_eval):
                print(f"Epoch {epoch}|{config.training['epochs']}, Batch {i_batch + 1}|{len_loader_eval}: "
                      f"Test Loss -- {loss_eval.item():.6f}, U -- {loss_u.item():.6f}, "
                      f"V -- {loss_v.item():.6f}, W -- {loss_w.item():.6f}, "
                      f"P -- {loss_p.item():.6f}, NMAE -- {nmae.item():.6f}, MAE -- {mae_pred.item():.6f}")
    
    return (epoch_test_loss, epoch_loss_u, epoch_loss_v, epoch_loss_w, epoch_loss_p,
            epoch_test_nmae, epoch_test_mae)


def main():
    parser = argparse.ArgumentParser(description='Distributed training entrypoint')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--model_type', type=str, default='dgcnn', choices=['dgcnn', 'carotid'],
                        help='Switch between generic DGCNN and carotid variant')
    args = parser.parse_args()
    
    # Bootstrap DDP
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    ddp_setup()
    
    # Load config
    config = Config.load(args.config)
    device = int(os.environ['LOCAL_RANK'])
    
    # Dataset factory
    if args.model_type == 'carotid':
        dataset_class = CarotidArtery_Centerline
    else:
        dataset_class = Data_Centerline_Graph
    
    train_loader_ob, train_loader_bo, valid_loader_ob, valid_loader_bo = create_dataloaders(
        config, dataset_class, device
    )
    
    # Model
    model = create_model(config, args.model_type, device)
    
    # Output path
    save_path = config.training['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Optimizer
    init_lr = config.training['init_lr']
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(save_path, "Board/"))
    
    # Resume if needed
    start_epoch = config.training['start_epoch']
    best_loss = 2.0
    if config.training.get('load_path'):
        loc = f'cuda:{device}'
        load_model = torch.load(config.training['load_path'], map_location=loc)
        weights = load_model['state']
        start_epoch = load_model['epoch'] + 1
        init_lr = load_model['learningrate']
        best_loss = load_model['loss']
        model.module.load_state_dict(weights)
        if device == 0:
            print(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config.training['epochs'] + 1):
        # Train
        epoch_loss, epoch_ob_u, epoch_ob_v, epoch_ob_w, epoch_ob_p, epoch_mae = train_epoch(
            model, train_loader_ob, train_loader_bo, optimizer, config, device, epoch
        )
        
        if device == 0:
            log = (f"Epoch {epoch}|{config.training['epochs']}, Train Loss -- {epoch_loss:.6f}, "
                   f"U -- {epoch_ob_u:.6f}, V -- {epoch_ob_v:.6f}, "
                   f"W -- {epoch_ob_w:.6f}, P -- {epoch_ob_p:.6f}, MAE -- {epoch_mae:.6f}")
            print(log)
            
            if epoch % config.training['save_interval'] == 0:
                write_log(save_path, log)
            
            # TensorBoard logging
            writer.add_scalars("Loss/Train", {
                'Total': epoch_loss, 'U': epoch_ob_u, 'V': epoch_ob_v,
                'W': epoch_ob_w, 'P': epoch_ob_p
            }, epoch)
            writer.add_scalar('MAE/Train', epoch_mae, epoch)
        
        # Validate
        if epoch % config.training['valid_interval'] == 0:
            (epoch_test_loss, epoch_loss_u, epoch_loss_v, epoch_loss_w, epoch_loss_p,
             epoch_test_nmae, epoch_test_mae) = validate_epoch(
                model, valid_loader_ob, valid_loader_bo, config, device, epoch
            )
            
            if device == 0:
                log = (f"Epoch {epoch}|{config.training['epochs']}, Test Loss -- {epoch_test_loss:.6f}, "
                       f"U -- {epoch_loss_u:.6f}, V -- {epoch_loss_v:.6f}, "
                       f"W -- {epoch_loss_w:.6f}, P -- {epoch_loss_p:.6f}, "
                       f"NMAE -- {epoch_test_nmae:.6f}, MAE -- {epoch_test_mae:.6f}")
                print(log)
                write_log(save_path, log)
                
                # TensorBoard logging
                writer.add_scalars("Loss/Valid", {
                    'Total': epoch_test_loss, 'U': epoch_loss_u, 'V': epoch_loss_v,
                    'W': epoch_loss_w, 'P': epoch_loss_p
                }, epoch)
                writer.add_scalar('Metrics/Valid', epoch_test_nmae, epoch)
                writer.add_scalar('MAE/Valid', epoch_test_mae, epoch)
                
                # Persist best checkpoint
                if epoch_test_loss < best_loss:
                    best_loss = epoch_test_loss
                    state = {
                        'state': model.module.state_dict(),
                        'epoch': epoch,
                        'learningrate': init_lr,
                        'loss': best_loss
                    }
                    torch.save(state, os.path.join(save_path, 'best_loss.pth'))
        
        # Periodic checkpointing
        if epoch % config.training['save_interval'] == 0 and device == 0:
            state = {
                'state': model.module.state_dict(),
                'epoch': epoch,
                'learningrate': init_lr,
                'loss': epoch_loss
            }
            torch.save(state, os.path.join(save_path, get_epoch_str(epoch, config.training['epochs']) + '.pth'))
    
    # Tear down
    destroy_process_group()


if __name__ == '__main__':
    main()

