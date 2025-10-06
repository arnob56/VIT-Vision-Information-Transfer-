model_ft = train_model_all_metrics(
    model=model_ft,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,  # can be None
    dataloaders=dataloaders,     # must contain 'train', 'val', 'test'
    device=device,
    num_epochs=25
)
