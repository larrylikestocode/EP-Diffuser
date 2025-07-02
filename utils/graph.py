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
from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import coalesce
from torch_geometric.utils import degree
from torch_geometric.utils import dense_to_sparse

def add_edges(
        from_edge_index: torch.Tensor,
        to_edge_index: torch.Tensor,
        from_edge_attr: Optional[torch.Tensor] = None,
        to_edge_attr: Optional[torch.Tensor] = None,
        replace: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from_edge_index = from_edge_index.to(device=to_edge_index.device, dtype=to_edge_index.dtype)
    mask = ((to_edge_index[0].unsqueeze(-1) == from_edge_index[0].unsqueeze(0)) &
            (to_edge_index[1].unsqueeze(-1) == from_edge_index[1].unsqueeze(0)))
    if replace:
        to_mask = mask.any(dim=1)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr[~to_mask], from_edge_attr], dim=0)
        to_edge_index = torch.cat([to_edge_index[:, ~to_mask], from_edge_index], dim=1)
    else:
        from_mask = mask.any(dim=0)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr, from_edge_attr[~from_mask]], dim=0)
        to_edge_index = torch.cat([to_edge_index, from_edge_index[:, ~from_mask]], dim=1)
    return to_edge_index, to_edge_attr


def merge_edges(
        edge_indices: List[torch.Tensor],
        edge_attrs: Optional[List[torch.Tensor]] = None,
        reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)


def complete_graph(
        num_nodes: Union[int, Tuple[int, int]],
        ptr: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        loop: bool = False,
        device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    if ptr is None:
        if isinstance(num_nodes, int):
            num_src, num_dst = num_nodes, num_nodes
        else:
            num_src, num_dst = num_nodes
        edge_index = torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device),
                                          torch.arange(num_dst, dtype=torch.long, device=device)).t()
    else:
        if isinstance(ptr, torch.Tensor):
            ptr_src, ptr_dst = ptr, ptr
            num_src_batch = num_dst_batch = ptr[1:] - ptr[:-1]
        else:
            ptr_src, ptr_dst = ptr
            num_src_batch = ptr_src[1:] - ptr_src[:-1]
            num_dst_batch = ptr_dst[1:] - ptr_dst[:-1]
        edge_index = torch.cat(
            [torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device),
                                  torch.arange(num_dst, dtype=torch.long, device=device)) + p
             for num_src, num_dst, p in zip(num_src_batch, num_dst_batch, torch.stack([ptr_src, ptr_dst], dim=1))],
            dim=0)
        edge_index = edge_index.t()
    if isinstance(num_nodes, int) and not loop:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.contiguous()


def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def unbatch(
        src: torch.Tensor,
        batch: torch.Tensor,
        dim: int = 0) -> List[torch.Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)

### added ###
def create_triu_block(start: int, 
                      end: int):
    mask = torch.zeros((end-start, end-start), dtype=bool)
    tril_idx = torch.triu_indices(end-start, end-start)
    mask[tril_idx[0], tril_idx[1]] = True
    return mask


def create_casual_edge_index(ptr,
                             edge_index: Optional[torch.Tensor] = None,
                             device: Optional[Union[torch.device, str]] = None)-> torch.Tensor:
    mask_new = torch.block_diag(*[create_triu_block(start, end) for start, end in zip(ptr[:-1], ptr[1:])])
    if edge_index is not None:
        mask_old = torch.zeros(mask_new.shape, dtype=mask_new.dtype)
        mask_old[edge_index[0], edge_index[1]] = True
        mask_new = mask_new & mask_old
    return dense_to_sparse(mask_new)[0].to(device)


def mask_ptr(ptr,
             mask) -> torch.Tensor:
    assert ptr[-1] == mask.shape[0]
    new_ptr = [0]
    for start, end in zip(ptr[:-1], ptr[1:]):
        new_ptr.append(torch.sum(mask[start:end]))
        
    new_ptr = torch.cumsum(torch.tensor(new_ptr), dim=0)
    
    return new_ptr.detach().long().to(ptr.device)


def batch_to_ptr(batch) -> torch.Tensor:
    max_idx = torch.max(batch)
    ptr = batch.new_zeros(max_idx+2)
    for i in torch.arange(max_idx+1):
        num_idx = torch.sum(batch==i).long()
        ptr[i+1] = num_idx

    return torch.cumsum(ptr, dim=-1)