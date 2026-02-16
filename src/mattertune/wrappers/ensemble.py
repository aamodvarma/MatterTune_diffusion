from __future__ import annotations

import logging

import contextlib
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import ase
import torch

from .property_predictor import _atoms_list_to_dataloader, _resolve_properties
from ..finetune.properties import PropertyConfig

if TYPE_CHECKING:
    from ..finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig


PropertyType = Literal["system", "atom"]

logger = logging.getLogger(__name__)


class MatterTuneEnsemblePredictor:
    """
    A wrapper class for deep ensemble prediction using multiple fine-tuned models.

    This class provides an interface to make ensemble predictions from a list of
    trained MatterTune models. It returns the ensemble mean, elementwise variance,
    and the individual model predictions.

    Notes on uncertainty definitions
    --------------------------------
    - Variance is computed elementwise across ensemble members.
    - An additional scalar `uncertainty` is provided per property:
      - For atom-wise vector outputs (e.g., forces with shape [N, 3]), uncertainty
        is the trace of the covariance for each atom, i.e., sum of component
        variances. This equals E[||f - mean(f)||^2] and is rotation invariant.
      - For matrix outputs (e.g., stress with shape [3, 3]), uncertainty is the
        Frobenius-norm variance, i.e., sum of component variances. This equals
        E[||S - mean(S)||_F^2] and is rotation invariant.
      - For scalar outputs, uncertainty equals variance.
    """

    def __init__(
        self,
        lightning_modules: Sequence[
            FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig]
        ],
    ):
        if len(lightning_modules) == 0:
            raise ValueError(
                "At least one model is required for ensemble prediction.")

        self.lightning_modules = list(lightning_modules)

        _validate_ensemble_models(self.lightning_modules)

    def predict(
        self,
        atoms_list: list[ase.Atoms],
        properties: Sequence[str | PropertyConfig] | None = None,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        devices: Sequence[str |
                          torch.device] | str | torch.device | None = None,
        parallel: bool | None = None,
        unbiased_variance: bool = False,
        return_member_predictions: bool = True,
        return_uncertainty: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Predict properties for a list of atomic systems using a deep ensemble.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            List of atomic systems to predict properties for.
        properties : Sequence[str | PropertyConfig] | None, optional
            Properties to predict. Can be specified as strings or PropertyConfig objects.
            If None, predicts all properties supported by the model.
        batch_size : int, optional
            Batch size for prediction. Defaults to 1.
        num_workers : int, optional
            Number of workers used by the dataloader. Defaults to 0.
        devices : Sequence[str | torch.device] | str | torch.device | None, optional
            Devices to run the ensemble members on. If a single device is provided,
            all models run sequentially on that device. If a list is provided, it
            must have length 1 or match the number of models. Defaults to using
            each model's current device.
        parallel : bool | None, optional
            Whether to run models in parallel when each model has its own device.
            If None, this is automatically enabled when devices are distinct and
            match the number of models.
        unbiased_variance : bool, optional
            Whether to compute an unbiased sample variance. Defaults to False.
        return_member_predictions : bool, optional
            Whether to include individual model predictions in the output. Defaults to True.
        return_uncertainty : bool, optional
            Whether to include the scalar uncertainty for each property. Defaults to True.

        Returns
        -------
        list[dict[str, Any]]
            One entry per structure with keys:
            - "mean": dict[str, torch.Tensor]
            - "variance": dict[str, torch.Tensor]
            - "uncertainty": dict[str, torch.Tensor] (if requested)
            - "members": list[dict[str, torch.Tensor]] (if requested)
        """
        if parallel:
            logger.warning("Parallel prediction enabled. Setting num_workers=0 to avoid dataloader issues.")
            num_workers = 0
        
        num_models = len(self.lightning_modules)
        if unbiased_variance and num_models < 2:
            raise ValueError(
                "unbiased_variance=True requires at least two ensemble members."
            )

        _resolve_properties(properties, self.lightning_modules[0].hparams)
        property_type_map = _property_type_map(
            self.lightning_modules[0].hparams)

        device_list = _normalize_devices(self.lightning_modules, devices)
        if parallel is None:
            parallel = _should_parallel(device_list)

        if parallel:
            _validate_parallel_devices(device_list, num_models)
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                futures = [
                    executor.submit(
                        _predict_with_model,
                        model,
                        atoms_list,
                        properties,
                        batch_size,
                        num_workers,
                        device,
                    )
                    for model, device in zip(self.lightning_modules, device_list)
                ]
                member_predictions = [future.result() for future in futures]
        else:
            member_predictions = [
                _predict_with_model(
                    model,
                    atoms_list,
                    properties,
                    batch_size,
                    num_workers,
                    device,
                )
                for model, device in zip(self.lightning_modules, device_list)
            ]

        num_structures = len(atoms_list)
        for i, preds in enumerate(member_predictions):
            if len(preds) != num_structures:
                raise ValueError(
                    "Mismatch in prediction length for model index "
                    f"{i}: expected {num_structures}, got {len(preds)}."
                )

        outputs: list[dict[str, Any]] = []
        for struct_idx in range(num_structures):
            struct_member_preds = [
                member_predictions[m][struct_idx] for m in range(num_models)
            ]

            property_keys = list(struct_member_preds[0].keys())
            _ensure_property_keys_consistent(
                property_keys, struct_member_preds)

            mean_dict: dict[str, torch.Tensor] = {}
            var_dict: dict[str, torch.Tensor] = {}
            uncertainty_dict: dict[str, torch.Tensor] = {}

            for prop_name in property_keys:
                stacked = _stack_property(struct_member_preds, prop_name)
                mean = stacked.mean(dim=0)
                var = stacked.var(dim=0, unbiased=unbiased_variance)

                mean_dict[prop_name] = mean
                var_dict[prop_name] = var

                if return_uncertainty:
                    prop_type = _get_property_type(
                        prop_name, property_type_map)
                    uncertainty_dict[prop_name] = _reduce_variance(
                        var, prop_type)

            output: dict[str, Any] = {
                "mean": mean_dict,
                "variance": var_dict,
            }
            if return_uncertainty:
                output["uncertainty"] = uncertainty_dict
            if return_member_predictions:
                output["members"] = struct_member_preds

            outputs.append(output)
        
        # All convert to np.ndarray for easier downstream use, especially with ASE and other libraries
        for output in outputs:
            output["mean"] = {k: v.numpy() for k, v in output["mean"].items()}
            output["variance"] = {k: v.numpy() for k, v in output["variance"].items()}
            if return_uncertainty:
                output["uncertainty"] = {k: v.numpy() for k, v in output["uncertainty"].items()}
            if return_member_predictions:
                output["members"] = [
                    {k: v.numpy() for k, v in member.items()}
                    for member in output["members"]
                ]

        return outputs



def _to_cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_cpu_tree(v) for v in x]
        return type(x)(t)
    return x

def _property_type_map(
    hparams: FinetuneModuleBaseConfig,
) -> dict[str, PropertyType]:
    mapping = {prop.name: prop.property_type() for prop in hparams.properties}
    mapping.setdefault("energies_per_atom", "atom")
    return mapping  # type: ignore


def _get_property_type(
    prop_name: str, property_type_map: dict[str, PropertyType]
) -> PropertyType:
    if prop_name == "energies_per_atom":
        return "atom"
    return property_type_map.get(prop_name, "system")


def _reduce_variance(var: torch.Tensor, prop_type: PropertyType) -> torch.Tensor:
    if prop_type == "atom":
        if var.ndim <= 1:
            return var
        return var.sum(dim=tuple(range(1, var.ndim)))

    if var.ndim == 0:
        return var
    return var.sum()


def _stack_property(
    member_predictions: list[dict[str, torch.Tensor]], prop_name: str
) -> torch.Tensor:
    try:
        return torch.stack([pred[prop_name] for pred in member_predictions], dim=0)
    except Exception as exc:  # pragma: no cover - defensive for shape mismatches
        shapes = [
            tuple(pred[prop_name].shape) if prop_name in pred else None
            for pred in member_predictions
        ]
        raise ValueError(
            f"Failed to stack property '{prop_name}' across ensemble members. "
            f"Shapes: {shapes}"
        ) from exc


def _ensure_property_keys_consistent(
    reference_keys: list[str],
    member_predictions: list[dict[str, torch.Tensor]],
):
    ref_set = set(reference_keys)
    for idx, pred in enumerate(member_predictions[1:], start=1):
        if set(pred.keys()) != ref_set:
            raise ValueError(
                "Ensemble members must return the same property keys. "
                f"Model index {idx} has keys {sorted(pred.keys())}, "
                f"expected {sorted(ref_set)}."
            )


def _validate_ensemble_models(
    models: Sequence[FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig]],
):
    reference = {
        prop.name: prop.property_type() for prop in models[0].hparams.properties
    }
    for idx, model in enumerate(models[1:], start=1):
        candidate = {prop.name: prop.property_type()
                     for prop in model.hparams.properties}
        if candidate != reference:
            raise ValueError(
                "All ensemble models must expose the same properties with the same "
                "types. Mismatch at index "
                f"{idx}: expected {reference}, got {candidate}."
            )


def _predict_with_model(
    model: FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig],
    atoms_list: list[ase.Atoms],
    properties: Sequence[str | PropertyConfig] | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    _resolve_properties(properties, model.hparams)
    model.eval()

    if device.type == "cuda":
        cuda_ctx = torch.cuda.device(device)
    else:
        cuda_ctx = contextlib.nullcontext()

    inference_ctx = (
        contextlib.nullcontext()
        if model.requires_disabled_inference_mode()
        else torch.inference_mode()
    )

    dataloader = _atoms_list_to_dataloader(
        atoms_list, model, batch_size=batch_size, num_workers=num_workers
    )

    predictions: list[dict[str, torch.Tensor]] = []
    with cuda_ctx, inference_ctx:
        model.to_device(device)
        for batch in dataloader:
            batch = model.batch_to_device(batch, device)
            batch_preds = model.predict_step(batch=batch, batch_idx=0)
            batch_preds = [_to_cpu_tree(p) for p in batch_preds]
            predictions.extend(batch_preds) # type: ignore

    if len(predictions) != len(atoms_list):
        raise ValueError(
            "Mismatch in predictions length. "
            f"Expected {len(atoms_list)}, got {len(predictions)}."
        )

    return predictions


def _normalize_devices(
    models: Sequence[FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig]],
    devices: Sequence[str | torch.device] | str | torch.device | None,
) -> list[torch.device]:
    if devices is None:
        return [_infer_model_device(model) for model in models]

    if isinstance(devices, (str, torch.device)):
        return [torch.device(devices) for _ in models]

    device_list = [torch.device(device) for device in devices]
    if len(device_list) == 1:
        return device_list * len(models)
    if len(device_list) != len(models):
        raise ValueError(
            "Length of devices must be 1 or match the number of models. "
            f"Got {len(device_list)} devices and {len(models)} models."
        )
    return device_list


def _infer_model_device(
    model: FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig],
) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _should_parallel(devices: Iterable[torch.device]) -> bool:
    device_list = list(devices)
    if len(device_list) <= 1:
        return False
    if len({(device.type, device.index) for device in device_list}) != len(device_list):
        return False
    return all(device.type == "cuda" for device in device_list)


def _validate_parallel_devices(devices: Sequence[torch.device], num_models: int):
    if len(devices) != num_models:
        raise ValueError(
            "Parallel prediction requires one device per model. "
            f"Got {len(devices)} devices for {num_models} models."
        )
    if len({(device.type, device.index) for device in devices}) != len(devices):
        raise ValueError(
            "Parallel prediction requires distinct devices for each model."
        )
