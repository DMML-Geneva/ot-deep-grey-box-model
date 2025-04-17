from torch.utils.data import DataLoader
from src.sampler.loaders import LoaderSampler


def get_dataset_loader(
    tr_x_dataset,
    tr_y_dataset,
    val_y_dataset=None,
    batch_size=1,
    val_batch_size=1,
    test_dataset=None,
    test_batch_size=-1,
    device="cuda",
):
    """
    Create the data loaders for the training, validation and test sets.
    """
    # Source/Label data - Training
    train_x_loader = None
    if tr_x_dataset is not None:
        # Create the train
        train_x_loader = LoaderSampler(
            DataLoader(tr_x_dataset, batch_size=batch_size, shuffle=True),
            device=device,
        )

    # Target/Unlabeled data
    train_y_loader = None
    if tr_y_dataset is not None:
        train_y_loader = LoaderSampler(
            DataLoader(tr_y_dataset, batch_size=batch_size, shuffle=True),
            device=device,
        )

    # Validation data
    val_y_loader = None
    if val_y_dataset is not None:

        val_y_loader = LoaderSampler(
            DataLoader(
                val_y_dataset, batch_size=val_batch_size, shuffle=False
            ),
            device=device,
        )

    # Create the test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = LoaderSampler(
            DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False
            ),
            device=device,
        )

    return (
        train_x_loader,
        train_y_loader,
        val_y_loader,
        test_loader,
    )


def get_data_loader(
    y_dataset=None, batch_size=1, shuffle=False, device="cuda"
):

    y_loader = LoaderSampler(
        DataLoader(y_dataset, batch_size=batch_size, shuffle=False),
        device=device,
    )
    return y_loader
