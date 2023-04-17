import torch.utils.data
from . import single_dataset

class CustomDatasetDataLoader(object):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, dataset_type, train, batch_size, 
		dataset_root="", transform_1=None, transform_2=None, classnames=None,
        num_workers=0, **kwargs):

        self.train = train
        self.dataset = getattr(single_dataset, dataset_type)()
        self.dataset.initialize(root=dataset_root, 
                        transform_1=transform_1, transform_2=transform_2, 
                        classnames=classnames, **kwargs)

        self.classnames = classnames
        self.batch_size = batch_size

        dataset_len = len(self.dataset)
        cur_batch_size = min(dataset_len, batch_size)
        assert cur_batch_size != 0, \
            'Batch size should be nonzero value.'

        if self.train:
            drop_last = True
            sampler = torch.utils.data.RandomSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, 
	    			self.batch_size, drop_last)
        else:
            drop_last = False
            sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, 
	    			self.batch_size, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         num_workers=int(num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


