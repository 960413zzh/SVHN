import torch
import torch.nn
from torch.nn import functional as F


def features_loss(activations, ema_activations):

    assert activations.size() == ema_activations.size()
    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    norm = torch.reshape(torch.norm(activations, 2, 1), (-1, 1))
    norm_similarity = activations / norm

    ema_norm = torch.reshape(torch.norm( ema_activations, 2, 1), (-1, 1))
    ema_norm_similarity = ema_activations / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


class Focal_Loss(torch.nn.Module):
    def __init__(self, gamma=1):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma


    def forward(self, preds, labels, batch_size):
        """
        preds:softmax输出结果
        labels:真实值
        """
        a_weights = []
        preds = F.softmax(preds, dim=1)
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))
        target = labels.view(y_pred.size())
        ce = -1 * torch.log(y_pred) * target
        ce_loss = -ce
        pt = torch.exp(ce_loss)

        threshold, _ = torch.max(preds, dim=1)
        for i in range(batch_size):
            a = pt[i]
            b = threshold[i]
            high_th_weight = torch.ones_like(a)
            low_th_weight = (torch.ones_like(a) - a) ** self.gamma
            weight = torch.where(a < b, high_th_weight, low_th_weight)
            a_weights.append(weight)
        weights = torch.tensor([item.cpu().detach().numpy() for item in a_weights]).cuda()
        floss = (weights*ce)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)