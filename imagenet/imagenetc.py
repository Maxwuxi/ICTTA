import logging

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
import math
import numpy as np
import tent
import norm
import cotta

from conf import cfg, load_cfg_fom_args
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)
def compute_metrics(model, x, y, num_classes=10, batch_size=100, device=None):
    if device is None:
        device = x.device

    all_preds = []
    all_labels = []

    acc = 0
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)

            output = model(x_curr)
            pred = output.argmax(dim=1)

            acc += (pred == y_curr).float().sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_curr.cpu().numpy())

    acc /= x.shape[0]

    
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'acc': {
            'acc': acc
        }
    }



def create_imbalanced_dataset(x, y, total_samples, num_classes):

    weights = np.array(range(5)) + 1
    samples_per_class = np.repeat(weights,200)

    selected_indices = []

    
    for cls in range(num_classes):
    
        indices = (y == cls).nonzero(as_tuple=True)[0].tolist()  
        n_samples = samples_per_class[cls]
        if len(indices)<1:
            continue
 
        selected_indices.extend(np.random.choice(indices, size=n_samples, replace=True).tolist())  

    np.random.shuffle(selected_indices)  

    
    new_x = x[selected_indices]
    new_y = y[selected_indices]

    return new_x, new_y




def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()

                
            x_test, y_test = create_imbalanced_dataset(x_test, y_test,cfg.CORRUPTION.NUM_EX,1000)
            metrics = compute_metrics(model, x_test, y_test,1000,cfg.TEST.BATCH_SIZE)
            #acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            #err = 1. - acc
            err = 1. -  metrics['acc']['acc']
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"Precision: {metrics['macro']['precision']:.2%}")
            logger.info(f"Recall: {metrics['macro']['recall']:.2%}")
            logger.info(f"F1: {metrics['macro']['f1']:.2%}")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')
