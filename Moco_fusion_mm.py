"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=4, feat_dim=4):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def flatten(t):
    return t.reshape(t.shape[0], -1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, query_encoder, key_encoder, dim=32, K=1024, m=0.999, T=0.05, mlp=True, feat_dim=1280, normalize=False, num_classes=4, modality='fusion', use_dim_matching_layer=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        modality: 'fusion', 'sensor', or 'vision'
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.modality = modality
        self.use_dim_matching_layer = use_dim_matching_layer
        
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = query_encoder
        self.encoder_k = key_encoder
        
        # 모달리티에 따라 linear layer 차원 조정
        if modality == 'fusion':
            if self.use_dim_matching_layer:
                self.feat_dim = feat_dim
            else:
                self.feat_dim = feat_dim+128
        else:
            self.feat_dim = feat_dim
        self.linear = nn.Linear(self.feat_dim, num_classes)
        self.linear_k = nn.Linear(self.feat_dim, num_classes)
        self.linear_add = nn.Linear(self.feat_dim, num_classes)
        self.Wstar_layer = nn.Linear(self.feat_dim, num_classes, bias=False)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.classifier.weight.shape[1]
            self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)).to(device)
            self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)).to(device)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_l", torch.randint(0, num_classes, (K, )))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -2
        self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = normalize

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]
            
            return children[-1][self.layer]
        return None

    def _hook_k(self, _, __, output):
        self.feat_after_avg_k = flatten(output)
        if self.normalize: 
           self.feat_after_avg_k = nn.functional.normalize(self.feat_after_avg_k, dim=1)


    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)


    def _register_hook(self):
        layer_k = self._find_layer(self.encoder_k)
        assert layer_k is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_k.register_forward_hook(self._hook_k)

        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # self.linear.to(torch.device('cpu'))
        self.linear.to(device)
        self.linear_k.to(device)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue

        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)        
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _train_fusion(self, im_q, sen_q, im_k, sen_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets, vision_feature, sensor_feature
        """
        # compute query features
        query, enc_feature, vision_feature, sensor_feature = self.encoder_q(im_q, sen_q)  # query: NxC, MLP output, enc_feature: encoder feature
        query = nn.functional.normalize(query, dim=1)

        # vision과 sensor 피쳐 추출
        if hasattr(self.encoder_q, 'vision_model') and hasattr(self.encoder_q, 'sensor_model'):
            with torch.no_grad():
                _, vision_feature = self.encoder_q.vision_model(im_q)
                _, sensor_feature = self.encoder_q.sensor_model(sen_q)
                # use_dim_matching_layer가 있을 때만 dimension adapter를 통과
                if hasattr(self.encoder_q, 'use_dim_matching_layer') and self.encoder_q.use_dim_matching_layer:
                    if hasattr(self.encoder_q, 'vision_dimension_adapter') and self.encoder_q.vision_dimension_adapter is not None:
                        vision_feature = self.encoder_q.vision_dimension_adapter(vision_feature)
                    if hasattr(self.encoder_q, 'sensor_dimension_adapter') and self.encoder_q.sensor_dimension_adapter is not None:
                        sensor_feature = self.encoder_q.sensor_dimension_adapter(sensor_feature)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k, _, _, _= self.encoder_k(im_k, sen_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # compute logits
        features = torch.cat((query, k, self.queue.clone().detach().to(device)), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach().to(device)), dim=0)
        
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        self.linear.to(device)
        self.linear_add.to(device)
        
        logits_q = self.linear(self.feat_after_avg_q)
        logits_cl = self.linear_add(self.feat_after_avg_q) # add linear classifier

        return features, target, logits_q, logits_cl, vision_feature, sensor_feature

    def _train_sensor(self, sen_q, sen_k, labels):
        """
        Input:
            sen_q: a batch of query sensor data
            sen_k: a batch of key sensor data
        Output:
            logits, targets, vision_feature, sensor_feature
        """
        # compute query features
        query, enc_feature = self.encoder_q(sen_q)  # query: NxC, MLP output, enc_feature: encoder feature
        query = nn.functional.normalize(query, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.encoder_k(sen_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # compute logits
        features = torch.cat((query, k, self.queue.clone().detach().to(device)), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach().to(device)), dim=0)
        
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        self.linear.to(device)
        self.linear_add.to(device)
        
        logits_q = self.linear(self.feat_after_avg_q)
        logits_cl = self.linear_add(self.feat_after_avg_q) # add linear classifier

        return features, target, logits_q, logits_cl, None, None

    def _train_vision(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets, vision_feature, sensor_feature
        """
        # compute query features
        query, enc_feature = self.encoder_q(im_q)  # query: NxC, MLP output, enc_feature: encoder feature
        query = nn.functional.normalize(query, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # compute logits
        features = torch.cat((query, k, self.queue.clone().detach().to(device)), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach().to(device)), dim=0)
        
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        self.linear.to(device)
        self.linear_add.to(device)
        
        logits_q = self.linear(self.feat_after_avg_q)
        logits_cl = self.linear_add(self.feat_after_avg_q) # add linear classifier

        return features, target, logits_q, logits_cl, None, None

    def _inference_fusion(self, im_q, sen_q):
        self.linear.to(device)
        self.linear_add.to(device)
        q, enc_feature = self.encoder_q(im_q, sen_q)
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q) # linear classifier logits       
        linear_logits = self.linear_add(self.feat_after_avg_q)
        logits_cl = linear_logits

        return encoder_q_logits, logits_cl #q

    def _inference_sensor(self, sen_q):
        self.linear.to(device)
        self.linear_add.to(device)
        q, enc_feature = self.encoder_q(sen_q)
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q) # linear classifier logits       
        linear_logits = self.linear_add(self.feat_after_avg_q)
        logits_cl = linear_logits

        return encoder_q_logits, logits_cl #q

    def _inference_vision(self, im_q):
        self.linear.to(device)
        self.linear_add.to(device)
        q, enc_feature = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q) # linear classifier logits       
        linear_logits = self.linear_add(self.feat_after_avg_q)
        logits_cl = linear_logits

        return encoder_q_logits, logits_cl #q

    def forward(self, im_q=None, sen_q=None, im_k=None, sen_k=None, labels=None):
        if self.training:
            if self.modality == 'fusion':
                return self._train_fusion(im_q, sen_q, im_k, sen_k, labels)
            elif self.modality == 'sensor':
                return self._train_sensor(sen_q, sen_k, labels)
            elif self.modality == 'vision':
                return self._train_vision(im_q, im_k, labels)
        else:
            if self.modality == 'fusion':
                return self._inference_fusion(im_q, sen_q)
            elif self.modality == 'sensor':
                return self._inference_sensor(sen_q)
            elif self.modality == 'vision':
                return self._inference_vision(im_q)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Single GPU version: just return the tensor as-is.
    """
    return tensor
