from .cosine_lr import CosineLRScheduler


def create_scheduler(config, optimizer):
    num_epochs = config['num_epochs']

    if getattr(config, 'lr_noise', None) is not None:
        lr_noise = getattr(config, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if config['sched'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(config, 'lr_cycle_mul', 1.),
            lr_min=config['min_lr'],
            decay_rate=config['decay_rate'],
            warmup_lr_init=config['warmup_lr'],
            warmup_t=config['warmup_epochs'],
            cycle_limit=getattr(config, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(config, 'lr_noise_pct', 0.67),
            noise_std=getattr(config, 'lr_noise_std', 1.),
            noise_seed=getattr(config, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + config['cooldown_epochs']
    else:
        raise NotImplementedError

    return lr_scheduler, num_epochs