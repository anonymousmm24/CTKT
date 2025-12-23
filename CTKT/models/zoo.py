import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy


class ExperienceAttention(nn.Module):
    """
    Experience Attention module composed of spatial and channel attention.
    It maintains a task-specific set of parameters (s_t, c_t, Phi_t) and
    returns both the transformed features and the regularization penalty
    that aligns consecutive tasks.
    """

    def __init__(self, emb_d, num_tokens, n_tasks, reg_strength=1.0):
        super().__init__()
        self.emb_d = emb_d
        self.num_tokens = num_tokens
        self.max_tasks = n_tasks
        self.reg_strength = reg_strength

        self.task_count = 0
        self.spatial_params = nn.ParameterList()
        self.channel_params = nn.ParameterList()
        self.phi_layers = nn.ModuleList()

        # identity buffer reused in the orthogonality-like penalty
        self.register_buffer("identity", torch.eye(self.emb_d))

        # pre-create parameter slots for all tasks for simplicity
        for _ in range(self.max_tasks):
            self._append_task_params()

    def _append_task_params(self):
        # spatial parameters broadcast across batch and channels
        s = nn.Parameter(torch.ones(1, self.num_tokens, 1))
        # channel parameters broadcast across batch and tokens
        c = nn.Parameter(torch.ones(1, 1, self.emb_d))
        # simple linear projection for phi_t initialized close to identity
        phi = nn.Linear(self.emb_d, self.emb_d)
        nn.init.eye_(phi.weight)
        nn.init.constant_(phi.bias, 0.0)

        self.spatial_params.append(s)
        self.channel_params.append(c)
        self.phi_layers.append(phi)

    def set_task(self, task_id):
        if task_id >= self.max_tasks:
            raise ValueError(f"Requested task_id {task_id} exceeds configured max_tasks {self.max_tasks}")
        self.task_count = task_id

    def forward(self, z_t, task_id=None, train=False):
        """
        Args:
            z_t: tensor of shape (B, tokens, emb_d)
            task_id: explicit task index. Defaults to the internal task counter.
            train: whether to compute regularization terms.
        Returns:
            Tuple (transformed features, regularization loss)
        """
        if task_id is None:
            task_id = self.task_count

        if task_id >= len(self.spatial_params):
            raise ValueError(f"Task id {task_id} out of range for ExperienceAttention.")

        s_t = self.spatial_params[task_id]
        c_t = self.channel_params[task_id]
        phi_t = self.phi_layers[task_id]

        # spatial attention
        z_s = z_t * s_t
        # channel attention followed by learnable transformation
        z_sc = z_s * c_t
        z_c = phi_t(z_sc) + z_s

        reg_loss = z_t.new_tensor(0.0)
        if train and self.reg_strength > 0 and task_id > 0:
            prev_idx = task_id - 1
            prev_phi = self.phi_layers[prev_idx].weight.detach()
            phi_weight = phi_t.weight
            eye = self.identity.to(phi_weight.device)
            gram = torch.matmul(phi_weight, prev_phi.t())
            reg_loss = reg_loss + torch.norm(gram - eye, p='fro') ** 2

            prev_c = self.channel_params[prev_idx].detach()
            prev_s = self.spatial_params[prev_idx].detach()
            reg_loss = reg_loss + torch.norm(c_t - prev_c, p='fro') ** 2
            reg_loss = reg_loss + torch.norm(s_t - prev_s, p='fro') ** 2
            reg_loss = reg_loss * self.reg_strength

        return z_c, reg_loss


class TaskPromptGenerator(nn.Module):
    """
    Task Prompt Generator that maintains learnable task queries, performs
    self-attention across tasks, cross-modal attention with semantic embeddings,
    and produces task prompts along with auxiliary losses.
    """

    def __init__(
        self,
        embed_dim,
        n_tasks,
        num_heads=8,
        text_dim=None,
        mlp_ratio=4.0,
        dropout=0.0,
        temperature=0.07,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tasks = n_tasks
        self.text_dim = text_dim if text_dim is not None else embed_dim
        self.temperature = temperature

        # learnable query vectors for each task
        self.query_params = nn.ParameterList()
        for _ in range(self.max_tasks):
            q = nn.Parameter(torch.zeros(1, embed_dim))
            init.trunc_normal_(q, std=0.02)
            self.query_params.append(q)

        # linear projections for Q, K, V
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        # self-attention across task queries
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # cross-modal attention with semantic embeddings
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_proj = nn.Linear(self.text_dim, embed_dim)

        # feed-forward network
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        self.task_count = 0
        self.semantic_bank = [None for _ in range(self.max_tasks)]

    def set_task(self, task_id):
        if task_id >= self.max_tasks:
            raise ValueError(f"Requested task_id {task_id} exceeds max_tasks {self.max_tasks}")
        self.task_count = task_id

    def process_task_count(self):
        if self.task_count + 1 < self.max_tasks:
            self.task_count += 1

    def set_semantic_embedding(self, task_id, embedding):
        """
        Stores semantic embeddings (e.g., text features) for a given task.
        Args:
            task_id: index of task
            embedding: tensor of shape (N, text_dim)
        """
        if task_id >= self.max_tasks:
            raise ValueError(f"Task id {task_id} out of range for semantic bank.")
        if embedding is None:
            self.semantic_bank[task_id] = None
        else:
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            self.semantic_bank[task_id] = embedding.detach()

    def _gather_queries(self, task_id):
        queries = []
        for idx in range(task_id + 1):
            q = self.query_params[idx]
            if idx == task_id:
                queries.append(q)
            else:
                queries.append(q.detach())
        return torch.cat(queries, dim=0)

    def _gather_semantics(self, task_id, override_embeddings=None, device=None):
        semantic_list = []
        positive_index = None
        total = 0
        for idx in range(task_id + 1):
            if override_embeddings is not None and idx < len(override_embeddings) and override_embeddings[idx] is not None:
                sem = override_embeddings[idx]
            else:
                sem = self.semantic_bank[idx]
            if sem is None:
                continue
            if sem.dim() == 1:
                sem = sem.unsqueeze(0)
            if device is not None:
                sem = sem.to(device)
            semantic_list.append(sem)
            if idx == task_id:
                positive_index = total + sem.shape[0] - 1
            total += sem.shape[0]
        if len(semantic_list) == 0:
            return None, None
        return torch.cat(semantic_list, dim=0), positive_index

    def forward(self, task_id=None, semantic_overrides=None, train=False):
        """
        Generates task prompt and auxiliary losses.
        Args:
            task_id: task index (defaults to internal counter)
            semantic_overrides: optional list of semantic embeddings per task
            train: flag to control loss computation
        Returns:
            p_task: generated prompt tensor of shape (1, embed_dim)
            losses: dict containing 'match' and 'contrast' loss terms
        """
        if task_id is None:
            task_id = self.task_count
        if task_id >= self.max_tasks:
            raise ValueError(f"Task id {task_id} exceeds max_tasks {self.max_tasks}")

        device = self.query_params[task_id].device

        q_current = self.query_params[task_id]
        past_queries = self._gather_queries(task_id).to(device)

        Q = self.w_q(q_current)
        K = self.w_k(past_queries)
        V = self.w_v(past_queries)

        attn_out, attn_weights = self.self_attn(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))
        attn_out = attn_out.squeeze(0)
        q_prime = q_current + self.dropout(attn_out)

        # prompt matching loss based on cosine similarity
        match_loss = Q.new_tensor(0.0)
        if train:
            key_norm = F.normalize(K, dim=-1)
            query_norm = F.normalize(Q, dim=-1)
            logits = torch.matmul(query_norm, key_norm.t()).unsqueeze(0)
            target = torch.tensor([task_id], device=device, dtype=torch.long)
            match_loss = F.cross_entropy(logits, target)

        semantic_feats, positive_index = self._gather_semantics(task_id, semantic_overrides, device=device)
        contrast_loss = Q.new_tensor(0.0)

        if semantic_feats is not None:
            semantic_feats = self.text_proj(semantic_feats)
            q_prime_norm = self.norm_self(q_prime)
            cross_out, _ = self.cross_attn(
                q_prime_norm.unsqueeze(0), semantic_feats.unsqueeze(0), semantic_feats.unsqueeze(0)
            )
            cross_out = cross_out.squeeze(0)
            q_double = q_prime + self.dropout(cross_out)

            q_double_norm = self.norm_cross(q_double)
            mlp_out = self.mlp(q_double_norm)
            p_task = q_double + mlp_out

            if train and positive_index is not None and semantic_feats.shape[0] > 1:
                p_norm = F.normalize(p_task, dim=-1)
                text_norm = F.normalize(semantic_feats, dim=-1)
                logits = torch.matmul(p_norm, text_norm.t()).unsqueeze(0) / self.temperature
                target = torch.tensor([positive_index], device=device, dtype=torch.long)
                contrast_loss = F.cross_entropy(logits, target)
            elif train and positive_index is not None:
                p_norm = F.normalize(p_task, dim=-1)
                text_norm = F.normalize(semantic_feats, dim=-1)
                logits = torch.matmul(p_norm, text_norm.t()).unsqueeze(0) / self.temperature
                target = torch.tensor([0], device=device, dtype=torch.long)
                contrast_loss = F.cross_entropy(logits, target)
        else:
            q_double = q_prime
            q_double_norm = self.norm_cross(q_double)
            mlp_out = self.mlp(q_double_norm)
            p_task = q_double + mlp_out

        losses = {'match': match_loss, 'contrast': contrast_loss}
        return p_task, losses


class CTKTPrompt(nn.Module):
    """
    Combined module implementing CTKT prompting with Experience Attention and
    Task Prompt Generation. It enriches token features and produces key/value
    prompts while accumulating auxiliary losses.
    """

    def __init__(self, emb_d, n_tasks, prompt_param, num_tokens, key_dim=768):
        super().__init__()
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self.key_d = key_dim
        self.num_tokens = num_tokens

        # parse hyper-parameters
        self.prompt_len = int(prompt_param[0]) if len(prompt_param) > 0 else 1
        reg_strength = float(prompt_param[1]) if len(prompt_param) > 1 else 1.0
        self.alpha = float(prompt_param[2]) if len(prompt_param) > 2 else 1.0
        self.beta = float(prompt_param[3]) if len(prompt_param) > 3 else 1.0
        self.gamma = float(prompt_param[4]) if len(prompt_param) > 4 else 1.0
        num_heads = int(prompt_param[5]) if len(prompt_param) > 5 else 8
        dropout = float(prompt_param[6]) if len(prompt_param) > 6 else 0.0
        temperature = float(prompt_param[7]) if len(prompt_param) > 7 else 0.07
        text_dim = int(prompt_param[8]) if len(prompt_param) > 8 else emb_d

        self.prompt_len = max(1, self.prompt_len)
        self.loss_layers = [0]
        self.prompt_layers = [0]
        self.task_count = 0

        self.experience = ExperienceAttention(emb_d, self.num_tokens, n_tasks, reg_strength=1.0)
        self.exp_reg_strength = reg_strength
        self.generator = TaskPromptGenerator(
            embed_dim=emb_d,
            n_tasks=n_tasks,
            num_heads=num_heads,
            text_dim=text_dim,
            dropout=dropout,
            temperature=temperature,
        )

        # map generated task prompt into key/value tensors
        self.key_proj = nn.Linear(emb_d, emb_d * self.prompt_len)
        self.value_proj = nn.Linear(emb_d, emb_d * self.prompt_len)

    def set_semantic_embedding(self, task_id, embedding):
        self.generator.set_semantic_embedding(task_id, embedding)

    def process_task_count(self):
        if self.task_count + 1 < self.n_tasks:
            self.task_count += 1
        self.experience.task_count = self.task_count
        self.generator.process_task_count()

    def forward(self, x_query, layer_idx, x_block, train=False, task_id=None, semantic_overrides=None):
        if task_id is None:
            task_id = self.task_count
        task_id = min(task_id, self.n_tasks - 1)
        self.task_count = task_id

        self.experience.set_task(task_id)
        self.generator.set_task(task_id)

        collect_loss = train and (layer_idx in self.loss_layers)
        z_enhanced, reg_loss = self.experience(x_block, task_id=task_id, train=collect_loss)
        if not collect_loss:
            reg_loss = x_block.new_zeros(1)

        loss_total = x_block.new_zeros(1)
        if collect_loss:
            loss_total = loss_total + self.gamma * self.exp_reg_strength * reg_loss

        x_block = z_enhanced
        p_return = None

        if layer_idx in self.prompt_layers:
            p_task, losses = self.generator(task_id=task_id, semantic_overrides=semantic_overrides, train=train)
            if train:
                loss_total = loss_total + self.alpha * losses['match'] + self.beta * losses['contrast']

            key = self.key_proj(p_task).view(1, self.prompt_len, self.emb_d)
            value = self.value_proj(p_task).view(1, self.prompt_len, self.emb_d)
            B = x_block.size(0)
            key = key.repeat(B, 1, 1)
            value = value.repeat(B, 1, 1)
            p_return = [key, value]

        return p_return, loss_total, x_block



class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'ctkt':
            num_tokens = zoo_model.pos_embed.shape[1]
            self.prompt = CTKTPrompt(768, prompt_param[0], prompt_param[1], num_tokens=num_tokens)
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)
