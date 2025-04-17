from src.data_loader.reactdiff_dataset import ReactionDiffusionDataset


class LoaderSampler:
    def __init__(self, data_loader, device="cuda"):
        self.device = device
        self.data_loader = data_loader
        self.it = iter(self.data_loader)

    def sample(self, size=-1):
        assert size <= self.data_loader.batch_size
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.data_loader)
            return self.sample(size)

        if self.data_loader.batch_size < size:
            return self.sample(size)

        if size > 0 and size < self.data_loader.batch_size:
            if isinstance(batch, dict):
                for k in batch:
                    batch[k] = batch[k][:size]
            else:
                if size > 0:
                    batch = batch[:size]

        if (
            isinstance(self.data_loader.dataset, ReactionDiffusionDataset)
            and batch["x"].shape[0] == 1
        ):
            for k in batch:
                batch[k] = batch[k].squeeze(0)

        return batch

    def __len__(self):
        return len(self.data_loader)

    def reset(self):
        self.it = iter(self.data_loader)
