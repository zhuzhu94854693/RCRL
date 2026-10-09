import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage import io
from skimage.filters.thresholding import threshold_otsu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import RCRL, UnsupervisedLoss, compute_handcraft_di
from dataset import HCD

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_1 = 42
random.seed(seed_1)
np.random.seed(seed_1)
torch.manual_seed(seed_1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_1)
    torch.cuda.manual_seed_all(seed_1)

import argparse

parser = argparse.ArgumentParser(
    description='RCRL',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--dataset', type=int, default=5) # need change 1-5
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_sample', type=int, default=100)
parser.add_argument('--network', type=str, default='RCRL')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--warmup_epoch', type=int, default=5)
parser.add_argument('--alpha_ramp_epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_residual', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--consistency_w', type=float, default=0.1)
parser.add_argument('--result_folder', type=str,
                    default=r'./result/Img17/RCRL_unspervised')
parser.add_argument('--root_dir', type=str,
                    default=r'./datasets')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
DATASET = args.dataset
DATASET_NAME = ['shuguang', 'Italy', 'yellow_C', 'yellow_D', 'Img17']

print('Dataset:', DATASET_NAME[DATASET - 1])
Net = args.network
Epoch = args.epoch
Num_sample = args.num_sample
print('Network:', Net)
print('Device:', device)
print('Random seed:', seed_1)

transforms_set = transforms.Compose([transforms.ToTensor()])

train_data = HCD(train=True, dataset_id=DATASET, transform=None,
                 p_tra='IRGMcS', repeat=200)
test_data = HCD(train=False, dataset_id=DATASET, p_tra='IRGMcS')

img_height = test_data.H
img_width = test_data.W
print(f'Image size: {img_height} x {img_width}')

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=1, shuffle=False)

model = RCRL(in_channels=3, feat_dim=32, patch_size=3, neighbor_size=7,
             use_multiscale_ss=True, ss_patch_sizes=(5, 7, 9),
             ss_dilations=(1, 2)).to(device)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'[Model] params = {n_params:.2f} M')
print(f'[Model] initial residual_scale = '
      f'{model.diff_pass.residual_scale.item():.6f}  (应=0)')

criterion = UnsupervisedLoss(w_pseudo=1.0, w_di=0.5, w_patch_self=0.3).to(device)

residual_param = model.diff_pass.residual_scale
other_params = [p for n, p in model.named_parameters()
                if 'residual_scale' not in n]

optimizer = torch.optim.Adam([
    {'params': other_params, 'lr': args.lr,
     'weight_decay': args.weight_decay},
    {'params': [residual_param], 'lr': args.lr_residual,
     'weight_decay': 0.0}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=Epoch, eta_min=1e-6)


def confusion_matrix(true_value, output_data):
    true_value = true_value.reshape(-1)
    output_data = output_data.reshape(-1)
    pred = (output_data > 0.5).float()
    true_value = (true_value > 0.5).float()
    tp = int(((pred == 1) & (true_value == 1)).sum())
    tn = int(((pred == 0) & (true_value == 0)).sum())
    fp = int(((pred == 1) & (true_value == 0)).sum())
    fn = int(((pred == 0) & (true_value == 1)).sum())
    return tp, tn, fp, fn


def evaluate(tp, tn, fp, fn):
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    oa = recall = precision = f1 = 0.0
    false_alarm = missing_alarm = ciou = uciou = miou = kc = 0.0
    total = tp + tn + fp + fn

    if total > 0:
        oa = (tp + tn) / total
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    if (recall + precision) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    if (tn + fp) > 0:
        false_alarm = fp / (tn + fp)
    if (tp + fn) > 0:
        missing_alarm = fn / (tp + fn)
    if (tp + fp + fn) > 0:
        ciou = tp / (tp + fp + fn)
    if (tn + fp + fn) > 0:
        uciou = tn / (tn + fp + fn)
    miou = (ciou + uciou) / 2

    if total > 0:
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2)
        if (1 - pe) != 0:
            kc = (po - pe) / (1 - pe)

    return oa, recall, precision, f1, false_alarm, missing_alarm, \
        ciou, uciou, miou, kc


def train_one_epoch(train_loader_arg, model_arg, criterion_arg,
                     optimizer_arg, epoch):
    model_arg.train()
    total_loss = 0.0
    n_batch = 0
    alpha_sum = 0.0
    debug_printed = False

    with tqdm(total=len(train_loader_arg),
              desc='Train Epoch #{}'.format(epoch + 1)) as t:
        for img_1, img_2, pseudo in train_loader_arg:
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            pseudo = pseudo.to(device)

            if epoch == 0 and not debug_printed:
                uniq, cnt = torch.unique(pseudo, return_counts=True)
                print(f'\n[debug] pseudo values: '
                      f'{dict(zip(uniq.tolist(), cnt.tolist()))}')
                debug_printed = True

            out = model_arg(img_1, img_2)
            loss = criterion_arg(out, pseudo)

            if args.consistency_w > 0:
                x1f = torch.flip(img_1, dims=[3])
                x2f = torch.flip(img_2, dims=[3])
                out_f = model_arg(x1f, x2f)
                cons = F.mse_loss(out['di_prob'],
                                  torch.flip(out_f['di_prob'],
                                             dims=[3]).detach())
                loss = loss + args.consistency_w * cons

            optimizer_arg.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_arg.parameters(), 5.0)
            optimizer_arg.step()

            total_loss += loss.item()
            n_batch += 1
            alpha_sum += out['residual_scale'].item()

            t.set_postfix({
                'lr': '%.5f' % optimizer_arg.param_groups[0]['lr'],
                'loss': '%.4f' % loss.detach().cpu().item(),
                'alpha': '%.3f' % out['residual_scale'].item()
            })
            t.update(1)

    avg_loss = total_loss / max(n_batch, 1)
    avg_alpha = alpha_sum / max(n_batch, 1)
    return avg_loss, avg_alpha


@torch.no_grad()
def test(test_loader_arg, model_arg, epoch):
    model_arg.eval()
    tp = tn = fp = fn = 0
    canvas_prob = np.zeros((img_height, img_width), dtype=np.float32)
    canvas_gt = np.zeros((img_height, img_width), dtype=np.float32)
    canvas_mask = np.zeros((img_height, img_width), dtype=np.float32)
    res_scale_val = None

    pbar = tqdm(total=len(test_loader_arg),
                desc='Test Epoch #{}'.format(epoch + 1))

    for img_1, img_2, label, i, j in test_loader_arg:
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        label = label.to(device)

        out = model_arg(img_1, img_2)
        prob = out['di_prob']
        res_scale_val = out['residual_scale'].item()

        t_, n_, f_, m_ = confusion_matrix(label, prob)
        tp += t_
        tn += n_
        fp += f_
        fn += m_

        prob_np = prob[0, 0].cpu().numpy()
        gt_np = label[0, 0].cpu().numpy()
        ii = int(i.item()) if torch.is_tensor(i) else int(i)
        jj = int(j.item()) if torch.is_tensor(j) else int(j)
        h, w = prob_np.shape
        i2 = min(ii + h, img_height)
        j2 = min(jj + w, img_width)
        hh = i2 - ii
        ww = j2 - jj

        if hh > 0 and ww > 0:
            canvas_prob[ii:i2, jj:j2] = prob_np[:hh, :ww]
            canvas_gt[ii:i2, jj:j2] = gt_np[:hh, :ww]
            canvas_mask[ii:i2, jj:j2] = 1.0

        current_f1 = evaluate(tp, tn, fp, fn)[3]
        pbar.set_postfix({'f1': '%.4f' % current_f1})
        pbar.update(1)

    pbar.close()

    oa, recall, precision, f1, fa, ma, ciou, uciou, miou, kc = \
        evaluate(tp, tn, fp, fn)

    print(f'[Test] OA={oa:.4f} F1={f1:.4f} KC={kc:.4f} '
          f'Prec={precision:.4f} Rec={recall:.4f} MIOU={miou:.4f} '
          f'alpha={res_scale_val:.4f}')

    metrics = dict(tp=tp, tn=tn, fp=fp, fn=fn, oa=oa, recall=recall,
                   precision=precision, f1=f1, false_alarm=fa,
                   missing_alarm=ma, ciou=ciou, uciou=uciou, miou=miou,
                   kc=kc, res_scale=res_scale_val)

    return metrics, canvas_prob, canvas_gt, canvas_mask


def save_best_results(diff_prob_2d, save_dir, gt=None, mask=None,
                       epoch=None, metrics=None):
    os.makedirs(save_dir, exist_ok=True)
    d = diff_prob_2d.copy()

    valid = d[mask > 0] if mask is not None else d.reshape(-1)
    dmin = float(valid.min()) if valid.size else 0.0
    dmax = float(valid.max()) if valid.size else 1.0

    if (dmax - dmin) > 1e-8:
        diff_nor = (d - dmin) / (dmax - dmin)
    else:
        diff_nor = d.copy()

    if mask is not None:
        diff_nor[mask == 0] = 0.0

    diff_u8 = (diff_nor * 255).astype(np.uint8)
    io.imsave(os.path.join(save_dir, 'best_DI.png'), diff_u8)

    try:
        if mask is not None:
            thre = threshold_otsu(diff_u8[mask > 0])
        else:
            thre = threshold_otsu(diff_u8)
    except Exception:
        thre = 128

    cm = (diff_u8 >= thre) * 255
    if mask is not None:
        cm[mask == 0] = 0
    cm = cm.astype(np.uint8)
    io.imsave(os.path.join(save_dir, 'best_CM.png'), cm)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(diff_u8, cmap='gray')
    axes[0].set_title('Difference Map')
    axes[0].axis('off')

    axes[1].imshow(cm, cmap='gray')
    axes[1].set_title(f'Change Map (Otsu thre={thre:.0f})')
    axes[1].axis('off')

    cm_colored = np.zeros((*cm.shape, 3), dtype=np.uint8)
    cm_colored[:, :, 0] = cm
    axes[2].imshow(cm_colored)
    axes[2].set_title('Change Map (Color)')
    axes[2].axis('off')

    if gt is not None:
        axes[3].imshow(gt * 255, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    else:
        axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_visualization.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    eval_path = os.path.join(save_dir, 'best_eval.txt')
    with open(eval_path, 'w') as f:
        f.write(f'Epoch = {epoch}\n')
        f.write(f'Dataset = {DATASET_NAME[DATASET - 1]}\n')
        f.write('GT is used for epoch-wise evaluation and '
                'best-F1 selection.\n\n')
        f.write('Confusion Matrix:\n')
        f.write(f'TP={metrics["tp"]} || TN={metrics["tn"]} || '
                f'FP={metrics["fp"]} || FN={metrics["fn"]}\n\n')
        f.write('Evaluation Metrics:\n')
        f.write(f'OA = {metrics["oa"]:.6f}\n')
        f.write(f'Kappa = {metrics["kc"]:.6f}\n')
        f.write(f'Precision = {metrics["precision"]:.6f}\n')
        f.write(f'Recall = {metrics["recall"]:.6f}\n')
        f.write(f'F1 = {metrics["f1"]:.6f}\n')
        f.write(f'False Alarm = {metrics["false_alarm"]:.6f}\n')
        f.write(f'Missing Alarm = {metrics["missing_alarm"]:.6f}\n')
        f.write(f'CIOU = {metrics["ciou"]:.6f}\n')
        f.write(f'UCIOU = {metrics["uciou"]:.6f}\n')
        f.write(f'MIOU = {metrics["miou"]:.6f}\n')
        f.write(f'Residual scale = {metrics["res_scale"]:.6f}\n')
        f.write(f'Otsu threshold = {thre:.6f}\n')


if __name__ == '__main__':
    save_dir = args.result_folder
    os.makedirs(save_dir, exist_ok=True)

    model_dir = os.path.join(save_dir, 'models')
    result_dir = os.path.join(save_dir, 'results')
    best_dir = os.path.join(result_dir, 'best')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    print('\n[SelfCheck] verifying data & hand DI ...')
    img_1_dbg, img_2_dbg, pseudo_dbg = next(iter(train_loader))
    uniq, cnt = torch.unique(pseudo_dbg, return_counts=True)
    print(f'  pseudo values: {dict(zip(uniq.tolist(), cnt.tolist()))}')

    img_1_dbg = img_1_dbg.to(device)
    img_2_dbg = img_2_dbg.to(device)
    hp, hl = compute_handcraft_di(img_1_dbg, img_2_dbg)
    print(f'  hand_prob mean={hp.mean():.4f} std={hp.std():.4f} '
          f'min={hp.min():.4f} max={hp.max():.4f}')

    with torch.no_grad():
        out_dbg = model(img_1_dbg, img_2_dbg)

    print(f'  di_prob mean={out_dbg["di_prob"].mean():.4f} '
          f'std={out_dbg["di_prob"].std():.4f}')
    print(f'  |di_prob - hand_prob| = '
          f'{(out_dbg["di_prob"] - out_dbg["hand_prob"]).abs().mean():.6e} '
          f'(初始应≈0)')

    loss_dbg = criterion(out_dbg, pseudo_dbg.to(device))
    print(f'  loss = {loss_dbg.item():.4f}')
    print('[SelfCheck] OK\n')

    log_path = os.path.join(save_dir, 'train_log.txt')
    with open(log_path, 'w') as f:
        f.write(f'Dataset: {DATASET_NAME[DATASET - 1]}\n')
        f.write(f'Network: {Net}\n')
        f.write(f'Epochs: {Epoch}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Random seed: {seed_1}\n')
        f.write(f'LR main: {args.lr} | residual: {args.lr_residual}\n')
        f.write(f'warmup_epoch: {args.warmup_epoch} | '
                f'alpha_ramp_epoch: {args.alpha_ramp_epoch}\n')
        f.write(f'consistency_w: {args.consistency_w}\n')
        f.write(f'Initial residual_scale: '
                f'{model.diff_pass.residual_scale.item():.6f}\n')
        f.write('GT is NOT used for loss or backpropagation.\n')
        f.write('GT is used for epoch-wise evaluation and '
                'best-F1 selection.\n')
        f.write('=' * 80 + '\n')

    f1_sequence = []
    alpha_sequence = []
    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(Epoch):
        if epoch < args.warmup_epoch:
            alpha_max = 0.1
            model.diff_pass.residual_scale.requires_grad = False
            for p in model.diff_pass.residual_head.parameters():
                p.requires_grad = False
        else:
            model.diff_pass.residual_scale.requires_grad = True
            for p in model.diff_pass.residual_head.parameters():
                p.requires_grad = True
            t = (epoch - args.warmup_epoch) / max(1, args.alpha_ramp_epoch)
            alpha_max = min(1.0, 0.1 + 0.9 * t)

        if hasattr(model, 'set_alpha_max'):
            model.set_alpha_max(alpha_max)

        avg_loss, avg_alpha = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch)
        scheduler.step()

        print(f'[Epoch {epoch + 1}] avg_loss={avg_loss:.4f} '
              f'avg_alpha={avg_alpha:.4f} alpha_max={alpha_max:.4f}')

        metrics, di_full, gt_full, mask_full = test(test_loader, model, epoch)
        f1_sequence.append(metrics['f1'])
        alpha_sequence.append(avg_alpha)

        with open(log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | '
                    f'alpha={avg_alpha:.6f} | '
                    f'alpha_max={alpha_max:.4f} | '
                    f'F1={metrics["f1"]:.4f} | '
                    f'OA={metrics["oa"]:.4f} | '
                    f'Kappa={metrics["kc"]:.4f}\n')

        current_f1 = metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1

            print('\n' + '=' * 70)
            print('[BEST UPDATE]')
            print(f'Best epoch = {best_epoch}')
            print(f'Best F1 = {best_f1:.6f}')
            print('=' * 70 + '\n')

            for filename in os.listdir(best_dir):
                filepath = os.path.join(best_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    import shutil
                    shutil.rmtree(filepath)

            best_model_path = os.path.join(best_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'residual_scale': model.diff_pass.residual_scale.item(),
                'metrics': metrics,
                'f1': metrics['f1'],
                'oa': metrics['oa'],
                'kappa': metrics['kc'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'args': vars(args)
            }, best_model_path)

            save_best_results(di_full, best_dir, gt=gt_full,
                              mask=mask_full, epoch=best_epoch,
                              metrics=metrics)

            best_info_path = os.path.join(best_dir, 'best_epoch.txt')
            with open(best_info_path, 'w') as f:
                f.write(f'Dataset = {DATASET_NAME[DATASET - 1]}\n')
                f.write(f'Best epoch = {best_epoch}\n')
                f.write(f'Best F1 = {best_f1:.6f}\n')
                f.write(f'OA = {metrics["oa"]:.6f}\n')
                f.write(f'Kappa = {metrics["kc"]:.6f}\n')
                f.write(f'Precision = {metrics["precision"]:.6f}\n')
                f.write(f'Recall = {metrics["recall"]:.6f}\n')
                f.write(f'Residual scale = {metrics["res_scale"]:.6f}\n')

    print('\nTraining completed!')
    print(f'Best epoch = {best_epoch}')
    print(f'Best F1 = {best_f1:.4f}')
    print(f'Final epoch F1 = {f1_sequence[-1]:.4f}')
    print(f'Best results saved to: {best_dir}')

    with open(log_path, 'a') as f:
        f.write('=' * 80 + '\n')
        f.write(f'Best epoch = {best_epoch}\n')
        f.write(f'Best F1 = {best_f1:.6f}\n')
        f.write(f'Final epoch F1 = {f1_sequence[-1]:.6f}\n')
        f.write(f'Best result directory = {best_dir}\n')
        f.write(f'Final residual_scale = '
                f'{model.diff_pass.residual_scale.item():.6f}\n')

    try:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        epochs = np.arange(1, len(f1_sequence) + 1)

        ax1.plot(epochs, f1_sequence, 'r-o', label='F1', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('F1', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        ax1.scatter([best_epoch], [best_f1], s=100, marker='*',
                    label='Best F1')

        ax2 = ax1.twinx()
        ax2.plot(epochs, alpha_sequence, 'b-s', label='alpha', linewidth=2)
        ax2.set_ylabel('residual_scale', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        plt.title(f'{DATASET_NAME[DATASET - 1]} - F1 & residual_scale')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curve.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f'Training curve saved to {save_dir}/training_curve.png')
    except Exception as e:
        print(f'Plot failed: {e}')