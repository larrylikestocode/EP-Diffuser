# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter
from torch_scatter import segment_csr

class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               ptr: Optional[torch.Tensor] = None,
               joint: bool = False) -> None:
        
        if joint:
            assert ptr != None

            pred, target, prob, valid_mask, ptr = valid_filter(pred, target, prob, valid_mask, keep_invalid_final_step = False, ptr=ptr)
            
            # TODO implement for top k
            l2_norm = torch.norm(pred[:, :, -1] -
                                   target[:, -1].unsqueeze(-2),
                                   p=2, dim=-1) *  valid_mask[:, -1].unsqueeze(-1) # [A,K]

            l2_norm = segment_csr(src=l2_norm, indptr=ptr, reduce='sum') # [B, K]
            num_agents_batch = segment_csr(src=torch.where(valid_mask[:, -1], 1.0, 0.0), indptr=ptr, reduce='sum') # [B]

            best_mode = l2_norm.argmin(dim=-1) # [B]
  
            self.sum += (l2_norm[torch.arange(l2_norm.shape[0]), best_mode] / num_agents_batch).sum()
            self.count += l2_norm.size(0)
        else:
            pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
            pred_topk, _ = topk(self.max_guesses, pred, prob)
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                   target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                   p=2, dim=-1).min(dim=-1)[0].sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count