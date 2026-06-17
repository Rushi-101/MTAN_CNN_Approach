import os
import torch
import torch.nn.functional as F
import numpy as np

"""
Define task metrics, loss functions and model trainer here.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / binary_mask.sum()

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / binary_mask.sum()

    return loss


def gradient_cosine_weights(task_losses, shared_ref, num_tasks=3, eps=1e-8):
    """
    Gradient Cosine Similarity (GCS) task balancing.

    Computes each task loss's gradient w.r.t. a shared activation tensor
    (NOT a parameter — this makes it compatible with torch.compile and
    fused backbones where inspecting individual parameter gradients is
    awkward). Tasks whose gradients are LESS aligned with the average
    direction of the other tasks' gradients are up-weighted, since they
    are at the highest risk of being starved or overwritten during the
    shared backward pass.

    Args:
        task_losses : list of K scalar loss tensors, still attached to the
                      autograd graph (do NOT call .item() before passing in)
        shared_ref  : a tensor that all task losses causally depend on,
                      with requires_grad=True (e.g. the model's last shared
                      decoder feature map, exposed as `model._shared_ref`)
        num_tasks   : K, number of tasks (3 for NYUv2: semantic/depth/normal)
        eps         : numerical stability for cosine similarity

    Returns:
        weights : torch.Tensor of shape [K], detached from the graph,
                  always sums to exactly K. Use as:
                      loss = sum(weights[i] * task_losses[i] for i in range(K))

    Cost: K extra autograd.grad() calls per training step (one per task),
    each using retain_graph=True so the main loss.backward() / scaler flow
    afterwards is unaffected. For K=3 this is a small, bounded overhead —
    the same order of cost that GradNorm pays for its gradient-based weighting.
    """
    grads = []
    for L in task_losses:
        g = torch.autograd.grad(L, shared_ref, retain_graph=True, create_graph=False)[0]
        grads.append(g.reshape(-1).float())  # upcast for stable cosine sim under autocast

    K = num_tasks
    cos_sim = torch.zeros(K, K, device=shared_ref.device)
    for i in range(K):
        for j in range(K):
            cos_sim[i, j] = F.cosine_similarity(
                grads[i].unsqueeze(0), grads[j].unsqueeze(0), eps=eps)

    # mean similarity to OTHER tasks (excludes the diagonal self-similarity of 1.0)
    mean_sim = (cos_sim.sum(dim=1) - 1.0) / max(K - 1, 1)

    # invert: tasks with lower (more conflicting) similarity get higher weight
    weights = K * torch.softmax(-mean_sim, dim=0)
    return weights

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / binary_mask.sum()).item(), \
           (torch.sum(rel_err) / binary_mask.sum()).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    # clamp before acos to avoid NaN from fp precision
    cos_sim = torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)
    # stay on GPU, convert once
    error = torch.acos(cos_sim)
    error_deg = torch.rad2deg(error)
    mean = error_deg.mean().item()
    median = error_deg.median().item()
    return (mean, median,
            (error_deg < 11.25).float().mean().item(),
            (error_deg < 22.5).float().mean().item(),
            (error_deg < 30.0).float().mean().item())


checkpoint_dir = '/kaggle/working'

def save_checkpoint(epoch, model, optimizer, scheduler, keep_last=3, every_n=25):
    if epoch % every_n != 0:
        return
    path = f'{checkpoint_dir}/mtan_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    print(f'Checkpoint saved → mtan_epoch_{epoch}.pth')

    # prune old checkpoints
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].replace('.pth', ''))
    )
    for old in ckpts[:-keep_last]:
        os.remove(os.path.join(checkpoint_dir, old))


"""
=========== Universal Multi-task Trainer =========== 
"""


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, start_epoch, total_epoch=200, scaler=None):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])
    for index in range(start_epoch,total_epoch):
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)

        acc_semantic_loss = 0.0
        acc_depth_loss    = 0.0
        acc_depth_abs     = 0.0
        acc_depth_rel     = 0.0
        acc_normal_loss   = 0.0
        acc_normal_mean   = 0.0
        acc_normal_med    = 0.0
        acc_normal_11     = 0.0
        acc_normal_22     = 0.0
        acc_normal_30     = 0.0

        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            optimizer.zero_grad()


            with torch.autocast(device_type='cuda'):
                train_pred, logsigma = multi_task_model(train_data)
                train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                            model_fit(train_pred[1], train_depth, 'depth'),
                            model_fit(train_pred[2], train_normal, 'normal')]
                
                if opt.weight == 'gcs':
                    # Gradient Cosine Similarity task balancing.
                    # Requires the model to expose `_shared_ref`, a shared
                    # activation tensor set during forward() (see
                    # EfficientSegNetV2.forward — uses g_decoder[-1][1]).
                    # autocast is disabled for the autograd.grad calls below
                    # since cosine similarity on fp16 gradients can be
                    # numerically unstable; the reference tensor and grads
                    # are upcast to fp32 internally by autograd in this block.
                    shared_ref = multi_task_model._shared_ref

                    gcs_weights = gradient_cosine_weights(train_loss, shared_ref, num_tasks=3)

                    loss = sum(gcs_weights[i] * train_loss[i] for i in range(3))
                elif opt.weight == 'equal' or opt.weight == 'dwa':
                    loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(3)])
                else:
                    loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            abs_err, rel_err = depth_error(train_pred[1], train_depth)
            n_mean, n_med, n_11, n_22, n_30 = normal_error(train_pred[2], train_normal)

            acc_semantic_loss += train_loss[0].item()
            acc_depth_loss    += train_loss[1].item()
            acc_depth_abs     += abs_err
            acc_depth_rel     += rel_err
            acc_normal_loss   += train_loss[2].item()
            acc_normal_mean   += n_mean
            acc_normal_med    += n_med
            acc_normal_11     += n_11
            acc_normal_22     += n_22
            acc_normal_30     += n_30

        avg_cost[index, 0]  = acc_semantic_loss / train_batch
        avg_cost[index, 3]  = acc_depth_loss    / train_batch
        avg_cost[index, 4]  = acc_depth_abs     / train_batch
        avg_cost[index, 5]  = acc_depth_rel     / train_batch
        avg_cost[index, 6]  = acc_normal_loss   / train_batch
        avg_cost[index, 7]  = acc_normal_mean   / train_batch
        avg_cost[index, 8]  = acc_normal_med    / train_batch
        avg_cost[index, 9]  = acc_normal_11     / train_batch
        avg_cost[index, 10] = acc_normal_22     / train_batch
        avg_cost[index, 11] = acc_normal_30     / train_batch
        # compute mIoU and acc
        avg_cost[index, 1], avg_cost[index, 2] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]
                             
                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

        save_checkpoint(index, multi_task_model, optimizer, scheduler)


