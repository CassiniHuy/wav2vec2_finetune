from transformers import TrainingArguments

default_train_args = {
    'group_by_length': True,
    'per_device_train_batch_size': 32,
    'evaluation_strategy': "steps",
    'num_train_epochs': 30,
    'fp16': True,
    'gradient_checkpointing': True,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 500,
    'learning_rate': 1e-4,
    'weight_decay': 0.005,
    'warmup_steps': 1000,
    'save_total_limit': 1,
}

def get_training_arguments(output_dir: str, **kwargs) -> TrainingArguments:
    train_args = TrainingArguments(output_dir=output_dir, **default_train_args)
    for k, v in kwargs:
        if hasattr(train_args, k) is False:
            raise KeyError(f'Unrecognizable argument: {k}')
        if v is not None:
            train_args.__dict__[k] = v
    return train_args
