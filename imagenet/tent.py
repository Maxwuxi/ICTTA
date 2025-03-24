from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def ourloss(x: torch.Tensor):
    gamma=1
    # conf_threshold=0.4
    conf_threshold=0.6
    a1=1
    a2=1
    output = x.softmax(1)
    log_output = x.log_softmax(1)
    # 计算最大类别置信度和次高置信度
    sorted_output = torch.sort(output, dim=1, descending=True)
    pred, sub_pred = sorted_output[0][:, 0], sorted_output[0][:, 1]
    # 计算熵
    entropy = -(output * log_output).sum(1)
    batch_size=x.shape[0]
    weight_list = []
    # 根据置信度和熵进行动态筛选，生成权重
    for i in range(batch_size):
        # 仅计算满足条件的样本权重
        # if pred[i] > conf_threshold and entropy[i] < entropy_threshold:
        if pred[i] > conf_threshold:
            w = (1-(pred[i] - sub_pred[i])) ** gamma  # 计算样本的权重
        else:
            w = 0  # 对于不满足条件的样本，设置权重为 0
        weight_list.append(w)

    # 转换为 Tensor
    weight = torch.tensor(weight_list).cuda()
    weight = weight.view(-1, 1)
    entropy=-a1*(x.softmax(1) * x.log_softmax(1)).sum(1)
    regularization=a2*((weight)*(x.softmax(1) * x.log_softmax(1))).sum(1)
    loss=entropy+regularization
    return loss.mean(0)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    # loss = softmax_entropy(outputs).mean(0)
    loss = ourloss(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

# from copy import deepcopy
# import numpy
# import torch
# import torch.nn as nn
# import torch.jit
# from collections import defaultdict
#
#
# class Tent(nn.Module):
#     """Tent adapts a model by entropy minimization during testing.
#
#     Once tented, a model adapts itself by updating on every forward.
#     """
#     def __init__(self, model, optimizer, steps=1, episodic=False, num_classes=10):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.steps = steps
#         self.num_classes = num_classes
#         assert steps > 0, "tent requires >= 1 step(s) to forward and update"
#         self.episodic = episodic
#         self.ema_freq = defaultdict(lambda: 1.0)
#         for cls in range(num_classes):
#             _ = self.ema_freq[cls]  # Force initialization
# #        self.ema_freq = defaultdict(lambda: 1.0)
#         self.model_state, self.optimizer_state = \
#             copy_model_and_optimizer(self.model, self.optimizer)
# #    def forward_and_adapt(x, model, optimizer):
#     def forward_and_adapt(self, x):
#         outputs = self.model(x)
#         pseudo_labels = outputs.argmax(dim=1)
#         # Update EMA frequencies
#         # self.update_ema_freq(pseudo_labels)
#         # print(xxx)
#
#         # Calculate weights
#         weights = self.calc_weights(pseudo_labels).detach()
#
#         # Compute loss
#         loss=-10*(((outputs.softmax(1) * outputs.log_softmax(1)).sum(1))* weights).mean()
#
# #        loss = (self.softmax_entropy(outputs) * weights).mean()
#
#         # loss.requires_grad = True
#         # for name, param in self.model.named_parameters():
#         #     print(f"Parameter {name}: requires_grad={param.requires_grad}")
#         loss.backward()
#         self.optimizer.step()
#         self.optimizer.zero_grad()
#
#         return outputs
#     def forward(self, x):
#         if self.episodic:
#             self.reset()
#
#         for _ in range(self.steps):
#             outputs = self.forward_and_adapt(x)
#
#         return outputs
#
#
#     def update_ema_freq(self, pseudo_labels):
#         """更新EMA频率的类方法"""
#         current_counts = torch.bincount(
#             pseudo_labels,
#             minlength=self.num_classes
#         ).cpu().numpy()
#
#         alpha = 0.5
#         for cls in range(self.num_classes):
#             count = current_counts[cls]
#             if count > 0:
#                 self.ema_freq[cls] = alpha * self.ema_freq[cls] + (1 - alpha) * count
#
#     def calc_weights(self, pseudo_labels):
#         """计算样本权重的类方法"""
#         total = sum(self.ema_freq.values())
#         class_weights = {
#             cls: total / (self.num_classes * (self.ema_freq[cls] + 1e-8))  # 添加极小值防止除零
#             for cls in self.ema_freq
#         }
#
#         batch_weights = torch.tensor(
#             [class_weights[cls.item()] for cls in pseudo_labels],
#             device=pseudo_labels.device
#         )
#         return batch_weights / batch_weights.sum() * len(batch_weights)
#
#     def reset(self):
#         if self.model_state is None or self.optimizer_state is None:
#             raise Exception("cannot reset without saved model/optimizer state")
#         load_model_and_optimizer(self.model, self.optimizer,
#                                  self.model_state, self.optimizer_state)
# #    @staticmethod
# #    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
# #        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
#
#
#
#
# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.
#
#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.
#
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names
#
#
# def copy_model_and_optimizer(model, optimizer):
#     """Copy the model and optimizer states for resetting after adaptation."""
#     model_state = deepcopy(model.state_dict())
#     optimizer_state = deepcopy(optimizer.state_dict())
#     return model_state, optimizer_state
#
#
# def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
#     """Restore the model and optimizer states from copies."""
#     model.load_state_dict(model_state, strict=True)
#     optimizer.load_state_dict(optimizer_state)
#
#
# def configure_model(model):
#     """Configure model for use with tent."""
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#     return model
#
#
# def check_model(model):
#     """Check model for compatability with tent."""
#     is_training = model.training
#     assert is_training, "tent needs train mode: call model.train()"
#     param_grads = [p.requires_grad for p in model.parameters()]
#     has_any_params = any(param_grads)
#     has_all_params = all(param_grads)
#     assert has_any_params, "tent needs params to update: " \
#                            "check which require grad"
#     assert not has_all_params, "tent should not update all params: " \
#                                "check which require grad"
#     has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
#     assert has_bn, "tent needs normalization for its optimization"
