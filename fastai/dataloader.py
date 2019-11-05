import torch, queue
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from .imports import *
from .core import *
import collections,sys,traceback,threading
from torch.utils.data.sampler import Sampler

string_classes = (str, bytes)


def get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin: batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin, half) for sample in batch]
    raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False, len_list=None):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
        self.transpose,self.transpose_y,self.pad_idx,self.half = transpose,transpose_y,pad_idx,half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = MinibatchRandomSampler(dataset, batch_size, len_list) if shuffle else MinibatchSequentialSampler(dataset, batch_size, len_list)
                # sampler = MinibatchRandomSampler(dataset, batch_size,
                #                                  len_list) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self): return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b)==ml: return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i,o in enumerate(b):
            if self.pre_pad: res[i, -len(o):] = o
            else:            res[i,  :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)): return self.jag_stack(batch)
        elif isinstance(b, (int, float)): return np.array(batch)
        elif isinstance(b, string_classes): return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:   res[0] = res[0].T
        if self.transpose_y: res[1] = res[1].T
        return res

    def __iter__(self):
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield get_tensor(batch, self.pin_memory, self.half)


class MinibatchRandomSampler(Sampler):
    """Samples elements randomly in minibatches of same dataset, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size=1, len_list=None):
        self.data_source, self.batch_size, self.len_list = data_source, batch_size, len_list
        self.len = len(self.data_source)
        self.len_fixed = False

    def __iter__(self):
        # return iter(torch.randperm(len(self.data_source)).long())
        if self.batch_size == 1:
            return iter(torch.randperm(len(self.data_source)).long())
        else:
            # permutations within datasets, assuming not being dropped and even oversampled to always have full mini-batches
            perms = torch.empty(0, dtype=torch.int64)
            start_ind = 0
            for i in range(len(self.len_list)):
                perm = torch.randperm(self.len_list[i])
                perm = torch.add(perm, start_ind)
                perms = torch.cat([perms, perm])
                rem = self.len_list[i] - (self.len_list[i] // self.batch_size) * self.batch_size
                if rem > 0:
                    addition = self.batch_size - rem
                    perm = torch.randperm(self.len_list[i])
                    perm = torch.add(perm, start_ind)
                    perms = torch.cat([perms, perm[: addition]])
                    if not self.len_fixed:
                        self.len += addition
                start_ind += self.len_list[i]
                # if rem > 0:
                #     self.len_list[i] += addition

            # permutations between mini-batches
            perms2 = torch.empty(0, dtype=torch.int64)
            count = self.len // self.batch_size
            perm = torch.randperm(count)
            for i in perm:
                perms2 = torch.cat(
                    [perms2, perms[torch.mul(i, self.batch_size): torch.mul(i + 1, (self.batch_size))]])

            self.len_fixed = True
            return iter(perms2)

    def __len__(self):
        return self.len


class MinibatchSequentialSampler(Sampler):
    """Samples elements in minibatches of same dataset sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size=1, len_list=None):
        self.data_source, self.batch_size, self.len_list = data_source, batch_size, len_list
        self.len = len(self.data_source)
        self.len_fixed = False

    def __iter__(self):
        # return iter(range(len(self.data_source)))
        if self.batch_size == 1:
            return iter(range(len(self.data_source)))
        else:
            # Sequentially oversampled to always have full mini-batches
            perms = []
            start_ind = 0
            for i in range(len(self.len_list)):
                perm = list(range(start_ind, start_ind + self.len_list[i]))
                perms = perms + perm
                rem = self.len_list[i] - (self.len_list[i] // self.batch_size) * self.batch_size
                if rem > 0:
                    addition = self.batch_size - rem
                    perm = list(range(start_ind, start_ind + addition))
                    perms = perms + perm
                    if not self.len_fixed:
                        self.len += addition
                start_ind += self.len_list[i]
                # if rem > 0:
                #     self.len_list[i] += addition
            self.len_fixed = True
            return iter(perms)

    def __len__(self):
        return self.len


# class MinibatchSampler(object):
#     """Wraps another sampler to yield a mini-batch of indices.
#
#     Args:
#         sampler (Sampler): Base sampler.
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``
#
#     Example:
#         >> list(BatchSampler(range(10), batch_size=3, drop_last=False))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
#         >> list(BatchSampler(range(10), batch_size=3, drop_last=True))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     """
#
#     def __init__(self, sampler, batch_size, drop_last):
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#     def __iter__(self):
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# class MinibatchRandomSampler(Sampler):
#     """Samples elements randomly in minibatches of same dataset, without replacement.
#
#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """
#
#     def __init__(self, data_source, batch_size=1, len_list=None):
#         self.data_source, self.batch_size, self.len_list = data_source, batch_size, len_list
#
#     def __iter__(self):
#         # return iter(torch.randperm(len(self.data_source)).long())
#         perms = torch.empty(0, dtype=torch.int64)
#         start_ind = 0
#         for i in range(len(self.len_list)):
#             perm = torch.randperm(self.len_list[i])
#             perm = torch.add(perm, start_ind)
#             perms = torch.cat([perms, perm])
#             start_ind += self.len_list[i]
#         return iter(perms)
#
#     def __len__(self):
#         return len(self.data_source)