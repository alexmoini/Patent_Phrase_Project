"""
In this python file I want to make a transformer trainer class that will be used to train a LLM.

Requirements:
Should be able to train any architecture
Should be able to train multiple task types
Should be able to train multiple datasets
Should have adequate logging capabilities thru tensorboard
Should be able to train on multiple GPUs
Should work with Amazon Sagemaker AND Local Machine
Will be a wrapper of transformers Trainer class AND our own training loops
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class LargeLanguageModelTrainer():
    """
    This class is a wrapper of the transformers Trainer class and pytorch lightning Trainer class.
    It will be used to train any LLM architecture on any LLM task type.
    """
    def __init__(self,
                 model: Union[pl.LightningModule, transformers.PreTrainedModel],
                 hyperparameters: Dict[str, Any],
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 callbacks: Union[List[pl.Callback], List[transformers.TrainerCallback]] = None,
                 logger=None#: Union[pl.loggers, transformers.TrainerLogger] = None,
                 ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks
        self.logger = logger

        # Ensure consistency within the initialization var types
        if isinstance(self.model, transformers.PreTrainedModel):
            assert isinstance(self.logger, transformers.TrainerLogger), \
                             "logger must be a transformers.TrainerLogger"
            assert isinstance(self.callbacks, List[transformers.TrainerCallback]), \
                             "callbacks must be a List[transformers.TrainerCallback]"
        elif isinstance(self.model, pl.LightningModule):
            # assert isinstance(self.logger, (pl.loggers.TensorBoardLogger)), \
            #                  "logger must be a pl.loggers"
            # assert isinstance(self.callbacks, List[pl.Callback]), \
            #                  "callbacks must be a List[pl.Callback]"
            pass

    def train_mlm(self):
        """
        This function will train a masked language model
        """
        if isinstance(self.model, transformers.PreTrainedModel):
            # Train with transformers Trainer class
            args = transformers.TrainingArguments(**self.hyperparameters)
            trainer = transformers.Trainer(
                model=self.model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                callbacks=self.callbacks,
                logger=self.logger,
            )
            trainer.train()

        elif isinstance(self.model, pl.LightningModule):
            # Train with pytorch lightning Trainer class
            raise NotImplementedError("Training with pytorch lightning is not implemented yet")
        else:
            raise ValueError("Model must be either a \
                              transformers.PreTrainedModel \
                              or a pl.LightningModule")

    def train_summarization(self):
        """
        This function will train a summarization model
        """
        if isinstance(self.model, transformers.PreTrainedModel):
            # Train with transformers Trainer class
            args = transformers.TrainingArguments(**self.hyperparameters)
            trainer = transformers.Trainer(
                model=self.model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                callbacks=self.callbacks,
                logger=self.logger,
            )
            trainer.train()
        elif isinstance(self.model, pl.LightningModule):
            # Train with pytorch lightning Trainer class
            raise NotImplementedError("Training with pytorch lightning is not implemented yet")
        else:
            raise ValueError("Model must be either a \
                              transformers.PreTrainedModel \
                              or a pl.LightningModule")
    def train_similarity(self):
        """
        This function will train a similarity model
        """
        if isinstance(self.model, transformers.PreTrainedModel):
            # Train with transformers Trainer class
            args = transformers.TrainingArguments(**self.hyperparameters)
            raise NotImplementedError("Training with transformers.Trainer is not implemented yet")
        elif isinstance(self.model, pl.LightningModule):
            # Train with pytorch lightning Trainer class
            trainer = pl.Trainer(callbacks=self.callbacks,
                                 logger=self.logger,
                                 accelerator=self.hyperparameters['device'])
            train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.hyperparameters['batch_size'],
                                            shuffle=True,
                                            num_workers=self.hyperparameters['num_workers'])
            eval_dataloader = DataLoader(self.eval_dataset,
                                            batch_size=self.hyperparameters['batch_size'],
                                            shuffle=False,
                                            num_workers=self.hyperparameters['num_workers'])
            trainer.fit(self.model, train_dataloader, eval_dataloader)
        else:
            raise ValueError("Model must be either a \
                              transformers.PreTrainedModel \
                              or a pl.LightningModule")
