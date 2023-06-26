import time
from utils.parser import parser
from utils.data import SleepDataLoader
from models.dann import DANN_Lightning, CNN3L_Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def main():
    args = parser()
    data_dir = 'data/'
    pl.seed_everything(42)
    data_loader = SleepDataLoader(data_dir)
    train_loader = data_loader.train(batch_size=args.batch_size)
    val_loader = data_loader.val(batch_size=args.batch_size)
    test_loader = data_loader.test(batch_size=args.batch_size)
    ncl_loader = data_loader.ncl(batch_size=args.batch_size)

    callbacks = [EarlyStopping(monitor='val_BinaryAUROC', patience=args.patience, mode='max', verbose=True),
                 ModelCheckpoint(monitor='val_BinaryAUROC', mode='max')]

    if args.model == 'cnn':
        run_name = 'CNN3L_' + time.strftime("%Y%m%d-%H%M%S")
        model = CNN3L_Lightning(lr=args.lr)

        logger = WandbLogger(project='sleep-stage-classification', name=run_name, log_model='all')
        logger.experiment.config.update(args)

        trainer = Trainer(accelerator='gpu', max_epochs=args.epochs, logger=logger, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, dataloaders=(test_loader, ncl_loader))

    elif args.model == 'dann':
        run_name = 'DANN_' + time.strftime("%Y%m%d-%H%M%S")
        model = DANN_Lightning(train_loader, ncl_loader, lr=args.lr)

        logger = WandbLogger(project='sleep-stage-classification', name=run_name, log_model='all')
        logger.experiment.config.update(args)

        trainer = Trainer(accelerator='gpu', max_epochs=args.epochs, logger=logger, callbacks=callbacks)
        trainer.fit(model, val_dataloaders=test_loader)
        trainer.test(model, dataloaders=(test_loader, ncl_loader))


if __name__ == '__main__':
    main()
