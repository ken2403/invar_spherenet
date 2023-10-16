from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from itertools import islice

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

import invarsphere
from invarsphere.data.dataset import GraphDataset
from invarsphere.model import InvarianceSphereNet
from invarsphere.nn.scaling.scale_factor import ScaleFactor

logger = logging.getLogger(__name__)
logger.info(f"{invarsphere.__version__}")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size: int = 64,
        num_workers: int = 6,
        exclude_keys: list[str] = ["key"],
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_workers")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exclude_keys = exclude_keys

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            exclude_keys=self.exclude_keys,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            exclude_keys=self.exclude_keys,
        )


class ModelModule(pl.LightningModule):
    def __init__(
        self,
        model,
        property_name: str,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.8,
        regress_forces: bool = False,
        rho: float = 0.999,
    ):
        super().__init__()
        self.model = model
        self.n_params = self.model.n_param
        self.hparams["n_params"] = self.n_params
        self.save_hyperparameters(ignore=["model"])

        self.property_name = property_name
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.regress_forces = regress_forces
        self.rho = rho

        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def _training_step_energy(self, batch, batch_idx):
        pred_e, _ = self(batch)
        mse_e = self.mse(pred_e, batch[self.property_name])

        return mse_e

    def _training_step_forces(self, batch, batch_idx):
        pred_e, pred_f = self(batch)
        mae_e = self.mae(pred_e, batch[self.property_name])

        mse_f = self.mse(pred_f, batch["forces"])

        all_loss = (1 - self.rho) * mae_e + self.rho * mse_f

        return all_loss

    def training_step(self, batch, batch_idx):
        if self.regress_forces:
            loss = self._training_step_forces(batch, batch_idx)
        else:
            loss = self._training_step_energy(batch, batch_idx)

        return loss

    def _validation_step_energy(self, batch, batch_idx):
        pred_e, _ = self(batch)
        mae_e = self.mae(pred_e, batch[self.property_name])

        return mae_e

    def _validation_step_forces(self, batch, batch_idx):
        pred_e, pred_f = self(batch)
        mae_f = self.mae(pred_f, batch["forces"])

        return mae_f

    def validation_step(self, batch, batch_idx):
        if self.regress_forces:
            mae = self._validation_step_forces(batch, batch_idx)
        else:
            mae = self._validation_step_energy(batch, batch_idx)

        return mae

    def predict_step(self, batch, batch_idx):
        out = self(batch)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sche = ReduceLROnPlateau(opt, patience=self.patience, factor=self.factor)

        return {
            "optimizer": opt,
            "lr_scheduler": sche,
            "monitor": "val/mae_f",
        }


def json2args(json_path: str) -> argparse.Namespace:
    """Convert json file to argparse.Namespace. args contains all the
    parameters below.

    seed: int
    data_dir: str
    split_file: str
    num_workers: int
    batch_size: int
    emb_size_atom: int
    emb_size_edge: int
    emb_size_rbf: int
    emb_size_cbf: int
    emb_size_sbf: int
    emb_triplet: int
    emb_quad: int
    n_blocks: int
    n_targets: int
    max_n: int
    max_l: int
    rbf_smooth: bool
    triplets_only: bool
    cutoff: float
    p: int
    n_residual_output: int
    max_z: int
    extensive: bool
    regress_forces: bool
    direct_forces: bool
    activation: str
    weight_init: str
    align_initial_weight: bool
    property_name: str
    lr: float
    patience: int
    factor: float
    rho: float
    device: str
    num_batches: int
    """
    import json

    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args


def main(cmd_args: argparse.Namespace):
    # region get args and set logger
    save_dir: str = cmd_args.save_dir
    json_pth: str = cmd_args.json_pth
    log_level: str = cmd_args.loglevel
    if log_level == "INFO":
        level = logging.INFO
    elif log_level == "DEBUG":
        level = logging.DEBUG
    elif log_level == "WARNING":
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(level=level)

    args: argparse.Namespace = json2args(json_pth)
    logger.info(args)
    # endregion

    # region set seed
    logger.info(f"seed: {args.seed}")
    seed_everything(args.seed)
    # endregion

    # region load dataset
    logger.info("load dataset")
    dataset = GraphDataset(
        save_dir=f"{args.data_dir}",
        inmemory=False,
    )
    with open(f"{args.split_file}", "rb") as f:
        inds: dict = pickle.load(f)

    tr = dataset[inds["train"]]
    val = dataset[inds["val"]]
    test = dataset[inds["test"]]
    logger.info(f"tr: {len(tr)}")
    logger.info(f"val: {len(val)}")
    logger.info(f"test: {len(test)}")
    # endregion

    # region module setting
    logger.info("data module setting")
    data_modu = DataModule(
        train_dataset=tr,
        val_dataset=val,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    logger.info("model module setting")
    model = InvarianceSphereNet(
        emb_size_atom=args.emb_size_atom,
        emb_size_edge=args.emb_size_edge,
        emb_size_rbf=args.emb_size_rbf,
        emb_size_cbf=args.emb_size_cbf,
        emb_size_sbf=args.emb_size_sbf,
        emb_triplet=args.emb_triplet,
        emb_quad=args.emb_quad,
        n_blocks=args.n_blocks,
        n_targets=args.n_targets,
        max_n=args.max_n,
        max_l=args.max_l,
        rbf_smooth=args.rbf_smooth,
        triplets_only=args.triplets_only,
        cutoff=args.cutoff,
        cutoff_net="envelope",
        cutoff_kwargs={"p": args.p},
        n_residual_output=args.n_residual_output,
        max_z=args.max_z,
        extensive=args.extensive,
        regress_forces=args.regress_forces,
        direct_forces=args.direct_forces,
        activation=args.activation,
        weight_init=args.weight_init,
        align_initial_weight=args.align_initial_weight,
        scale_file=None,
    )

    model_modu = ModelModule(
        model=model,
        property_name=args.property_name,
        lr=args.lr,
        patience=args.patience,
        factor=args.factor,
        regress_forces=args.regress_forces,
        rho=args.rho,
    )
    model_modu.to(args.device)
    logger.info(model_modu.model)
    # endregion

    # region detect fitted/unfitted factors
    # recursively go through the submodules and get the ScaleFactor modules
    scale_factors: dict[str, ScaleFactor] = {
        name: module for name, module in model.named_modules() if isinstance(module, ScaleFactor)
    }
    fitted_scale_factors = [
        f"{name}: {module.scale_factor.item():.3f}" for name, module in scale_factors.items() if module.fitted
    ]
    unfitted_scale_factors = [name for name, module in scale_factors.items() if not module.fitted]
    fitted_scale_factors_str = ", ".join(fitted_scale_factors)
    logging.info(f"Fitted scale factors: [{fitted_scale_factors_str}]")
    unfitted_scale_factors_str = ", ".join(unfitted_scale_factors)
    logging.info(f"Unfitted scale factors: [{unfitted_scale_factors_str}]")
    # endregion

    # region reset all scale factors
    logging.info("Fitting all scale factors.")
    for name, scale_factor in scale_factors.items():
        if scale_factor.fitted:
            logging.info(f"{name} is already fitted in the checkpoint, resetting it. {scale_factor.scale_factor}")
        scale_factor.reset_()
        # endregion

    # region we do a single pass through the network to get the correct execution order of the scale factors
    scale_factor_indices: dict[str, int] = {}
    max_idx = 0

    # initialize all scale factors
    for name, module in scale_factors.items():

        def index_fn(name: str = name) -> None:
            nonlocal max_idx
            assert name is not None
            if name not in scale_factor_indices:
                scale_factor_indices[name] = max_idx
                logging.debug(f"Scale factor for {name} = {max_idx}")
                max_idx += 1

        module.initialize_(index_fn=index_fn)

    # single pass through network
    with torch.no_grad():
        one = next(iter(data_modu.val_dataloader()))
        one.to(args.device)
        model_modu.validation_step(one, 0)

    # sort the scale factors by their computation order
    sorted_factors = sorted(
        scale_factors.items(),
        key=lambda x: scale_factor_indices.get(x[0], math.inf),
    )

    logging.info("Sorted scale factors by computation order:")
    for name, _ in sorted_factors:
        logging.info(f"{name}: {scale_factor_indices[name]}")
        # endregion

    # region loop over the scale factors in the computation order
    # and fit them one by one
    logging.info("Start fitting ...")

    for name, module in sorted_factors:
        logging.info(f"Fitting {name}...")
        with module.fit_context_():
            for batch in islice(data_modu.val_dataloader(), args.num_batches):
                with torch.no_grad():
                    batch.to(args.device)
                    model_modu.validation_step(batch, 0)
            stats, ratio, value = module.fit_()

            logging.info(
                f"Variable: {name}, "
                f"Var_in: {stats['variance_in']:.3f}, "
                f"Var_out: {stats['variance_out']:.3f}, "
                f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            )

    # make sure all scale factors are fitted
    for name, module in sorted_factors:
        assert module.fitted, f"{name} is not fitted"
    # endregion

    # region save the fitted scale factors
    scale_dict: dict[str, float] = {name: module.scale_factor.item() for name, module in scale_factors.items()}
    with open(save_dir + "/scale_factors.json", "wb") as f:
        json.dump(scale_dict, f, indent=4)  # type: ignore
    logger.info(f"Saved scale factors to {save_dir}/scale_factors.json")
    # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str, help="the save directory name.")
    parser.add_argument("json_pth", type=str, help="the json path.")
    parser.add_argument("--loglevel", type=str, default="INFO", help="the log level.")
    args = parser.parse_args()
    main(args)
