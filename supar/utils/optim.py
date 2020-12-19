from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from typing import List, Optional
from supar.models import MultiBiaffineDependencyModel


class MultiTaskOptimizer():
    """An optimizer for MultiTaskModels. This abstracts away the complexities of 
        maintaining multiple optimizers for different parts of the model behind
        a single class which behaves much like any other optimizer class.

        Args:
            model (MultiBiaffineDependencyModel): The Multi Task Model which is
            supposed to be optimized
            task_names (List[str]): List of tasks supported by the model
            type (str): single/multiple.
            lr (float): learning rate
            mu (float): [description]
            nu (float): [description]
            epsilon (float): [description]

        Raises:
            Exception: [description]
    """
    def __init__(self, model: MultiBiaffineDependencyModel, task_names: List[str],
                 optimizer_type: str, lr: float, mu: float, nu: float,
                 epsilon: float, **kwargs) -> None:
        self.optim_type = optimizer_type
        if optimizer_type == 'single':
            self.optimizer = Adam(model.parameters(), lr, (mu, nu), epsilon)
        elif optimizer_type == 'multiple':
            self.optimizers = {
                task_name: Adam(model.get_task_specific_parameters(task_name), lr,
                                (mu, nu), epsilon)
                for task_name in task_names
            }
            self.optimizers['shared'] = Adam(model.get_shared_parameters(), lr,
                                             (mu, nu), epsilon)
        else:
            raise Exception(f"Unsupported optimizer type: {optimizer_type}")
        self.optim_list = []

    def set_mode(self, task_names: List[str], mode: str = 'train'):
        if isinstance(task_names, str):
            task_names = task_names.split(',')
        if self.optim_type == 'multiple':
            if mode == 'train':
                self.optim_list = ['shared'] + task_names
            elif mode == 'finetune':
                self.optim_list = task_names
            else:
                raise Exception(f"Unsupported mode: {mode}."
                                f"Only train/finetune are supported")

    def step(self) -> None:
        if self.optim_type == 'single':
            self.optimizer.step()
        else:
            for task_name in self.optim_list:
                self.optimizers[task_name].step()

    def zero_grad(self):
        if self.optim_type == 'single':
            self.optimizer.zero_grad()
        else:
            for task_name in self.optim_list:
                self.optimizers[task_name].zero_grad()


class MultiTaskScheduler():
    def __init__(self, optimizer: MultiTaskOptimizer, task_names: List[str],
                 optimizer_type: str, decay: float, decay_steps: float,
                 **kwargs: float) -> None:
        self.sched_type = optimizer_type
        if optimizer_type == 'single':
            self.scheduler = ExponentialLR(optimizer.optimizer,
                                           decay**(1 / decay_steps))
        else:
            self.schedulers = {
                task_name: ExponentialLR(optimizer, decay**(1 / decay_steps))
                for task_name, optimizer in optimizer.optimizers.items()
            }
        self.sched_list = []

    def set_mode(self, task_names: List[str], mode: str = 'train'):
        if isinstance(task_names, str):
            task_names = task_names.split(',')
        if self.sched_type == 'multiple':
            if mode == 'train':
                self.sched_list = ['shared'] + task_names
            elif mode == 'finetune':
                self.sched_list = task_names
            else:
                raise Exception(f"Unsupported mode: {mode}."
                                f"Only train/finetune are supported")

    def step(self) -> None:
        if self.sched_type == 'single':
            self.scheduler.step()
        else:
            for task_name in self.sched_list:
                self.schedulers[task_name].step()

    def get_last_lr(self, task_name: Optional[str] = None) -> float:
        if self.sched_type == 'single':
            return self.scheduler.get_last_lr()
        else:
            if task_name:
                return self.schedulers[task_name].get_last_lr()
            else:
                return self.schedulers[self.sched_list[-1]].get_last_lr()