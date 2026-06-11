from __future__ import annotations
import math

import contextlib
import importlib.util
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, cast

from ase import Atoms
import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch import config as functorch_config
from typing_extensions import assert_never, final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message, neighbor_list_and_relative_vec
from ..util import voigt_6_to_full_3x3_stress_torch

if TYPE_CHECKING:
    from orb_models.forcefield.base import AtomGraphs  # type: ignore[reportMissingImports] # noqa


log = logging.getLogger(__name__)


class ORBSystemConfig(C.Config):
    """Config controlling how to featurize a system of atoms."""

    radius: float
    """The radius for edge construction."""
    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""

    def _to_orb_system_config(self):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.atomic_system import SystemConfig  # type: ignore[reportMissingImports] # noqa

        return SystemConfig(
            radius=self.radius,
            max_num_neighbors=self.max_num_neighbors,
        )


@backbone_registry.register
class ORBBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["orb"] = "orb"
    """The type of the backbone."""

    pretrained_model: str
    """The name of the pretrained model to load."""

    system: ORBSystemConfig = ORBSystemConfig(radius=6.0, max_num_neighbors=120)
    """The system configuration, controlling how to featurize a system of atoms."""

    freeze_backbone: bool = False
    """Whether to freeze the backbone model."""

    checkpoint_path: str | None = None
    """Custom checkpoint path"""

    @override
    def create_model(self):
        return ORBBackboneModule(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the orb_models module is available
        if importlib.util.find_spec("orb_models") is None:
            raise ImportError(
                "The orb_models module is not installed. Please install it by running"
                ' pip install "orb_models@git+https://github.com/nimashoghi/orb_models.git"'
            )
            # NOTE: The 0.4.0 version of `orb_models` doesn't actually fully respect
            #   the `device` argument. We have a patch to fix this, and we have
            #   a PR open to fix this upstream. Until that is merged, users
            #   will need to install the patched version of `orb_models` from our fork:
            #   `pip install "orb_models@git+https://github.com/nimashoghi/orb_models.git"`
            #   PR: https://github.com/orbital-materials/orb_models/pull/35
            # FIXME: Remove this note once the PR is merged.

        # # Make sure pynanoflann is available
        # if importlib.util.find_spec("pynanoflann") is None:
        #     raise ImportError(
        #         "The pynanoflann module is not installed. Please install it by running"
        #         'pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"'
        #     )


@final
class ORBBackboneModule(
    FinetuneModuleBase["AtomGraphs", "AtomGraphs", ORBBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return ORBBackboneConfig

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _create_output_head(self, prop: props.PropertyConfig, pretrained_model):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.forcefield_heads import (
                ForceHead,
                NoiseHead,
                StressHead,
            )
            if self.hparams.using_partition:
                from orb_models.forcefield.forcefield_heads import GraphHeadPoolAfter, EnergyHeadPoolAfter
            else:
                from orb_models.forcefield.forcefield_heads import EnergyHead, GraphHead

        match prop:
            case props.NoisePropertyConfig():
                return NoiseHead(
                    latent_dim=256,
                    num_mlp_layers=1,
                    mlp_hidden_dim=256,
                )

            case props.EnergyPropertyConfig():
                if not self.hparams.reset_output_heads:
                    return pretrained_model.graph_head
                else:
                    if self.hparams.using_partition:
                        return EnergyHeadPoolAfter( # type: ignore[reportUnboundType] # noqa
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            predict_atom_avg = False,
                        )
                    else:
                        return EnergyHead( # type: ignore[reportUnboundType] # noqa
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            predict_atom_avg = False,
                        )

            case props.ForcesPropertyConfig():
                self.include_forces = True

                if prop.conservative:
                    return None
                else:
                    if not self.hparams.reset_output_heads:
                        return pretrained_model.node_head
                    else:
                        return ForceHead(
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            remove_mean=False,
                            remove_torque_for_nonpbc_systems=False,
                        )

            case props.StressesPropertyConfig():
                self.include_stress = True
                if prop.conservative:
                    return None
                else:
                    if not self.hparams.reset_output_heads:
                        return pretrained_model.stress_head
                    else:
                        return StressHead(
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                        )

            case props.GraphPropertyConfig():
                with optional_import_error_message("orb_models"):
                    from orb_models.forcefield.property_definitions import (  # type: ignore[reportMissingImports] # noqa
                        PropertyDefinition,
                    )
                if not self.hparams.reset_output_heads:
                    raise ValueError(
                        "Pretrained model does not support general graph properties, only energy, forces, and stresses are supported."
                    )
                else:
                    if self.hparams.using_partition:
                        return GraphHeadPoolAfter(  # type: ignore[reportUnboundType] # noqa
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            target=PropertyDefinition(
                                name=prop.name,
                                dim=1,
                                domain="real",
                            ),
                            node_aggregation=prop.reduction, # type: ignore
                        )
                    else:
                        return GraphHead(  # type: ignore[reportUnboundType] # noqa
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            target=PropertyDefinition(
                                name=prop.name,
                                dim=1,
                                domain="real",
                                row_to_property_fn= lambda l: torch.randn((1, 1))
                            ),
                            node_aggregation=prop.reduction, # type: ignore
                        )
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for ORB"
                    "Please ask the maintainers of ORB for support"
                )

    @override
    def create_model(self):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
            from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor

        # Get the pre-trained backbone
        # Load the pre-trained model from the ORB package
        if (
            pretrained_model_fn := pretrained.ORB_PRETRAINED_MODELS.get(
                self.hparams.pretrained_model
            )
        ) is None:
            raise ValueError(
                f"Unknown pretrained model: {self.hparams.pretrained_model}"
            )
        # We load on CPU here as we don't have a device yet.
        pretrained_model = pretrained_model_fn(device="cpu", compile=False)
        # This should never be None, but type checker doesn't know that so we need to check.
        assert pretrained_model is not None, "The pretrained model is not available"

        # This should be a `GraphRegressor` object, so we need to extract the backbone.
        assert isinstance(pretrained_model, DirectForcefieldRegressor) or isinstance(pretrained_model, ConservativeForcefieldRegressor), (
            f"Expected a GraphRegressor object, but got {type(pretrained_model)}"
        )
        if isinstance(pretrained_model, DirectForcefieldRegressor):
            self.conservative = False
        else:
            self.conservative = True
        backbone = pretrained_model.model

        if hasattr(self.hparams, 'checkpoint_path') and self.hparams.checkpoint_path:
            log.info(f"Loading custom checkpoint from: {self.hparams.checkpoint_path}")
            print(f"Loading custom checkpoint from: {self.hparams.checkpoint_path}", flush=True)
            checkpoint = torch.load(self.hparams.checkpoint_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            if hasattr(self, '_filter_state_dict'):
                state_dict = self._filter_state_dict(state_dict, backbone)
            
            # Load the state dict
            missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                log.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                log.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        # By default, ORB runs the `load_model_for_inference` function on the model,
        #   which sets the model to evaluation mode and freezes the parameters.
        #   We don't want to do that here, so we'll have to revert the changes.
        for param in backbone.parameters():
            param.requires_grad = True

        backbone = backbone.train()
        self.backbone = backbone

        self.system_config = self.hparams.system._to_orb_system_config()
        
        log.info(
            f'Loaded the ORB pre-trained model "{self.hparams.pretrained_model}". The model '
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        self.include_forces = False
        self.include_stress = False
        for prop in self.hparams.properties:
            head = self._create_output_head(prop, pretrained_model)
            # assert head is not None, (
            #     f"Find the head for the property {prop.name} is None"
            # )
            self.output_heads[prop.name] = head

    @override
    def trainable_parameters(self):
        if not self.hparams.freeze_backbone:
            yield from self.backbone.named_parameters()
        for head in self.output_heads.values():
            if head is not None:
                yield from head.named_parameters()
                
    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with contextlib.ExitStack() as stack:
            if self.conservative:
                stack.enter_context(torch.enable_grad())
                functorch_config.donated_buffer = False

            vectors, stress_displacement, generator = (
                data.compute_differentiable_edge_vectors()
            )
            assert stress_displacement is not None
            assert generator is not None
            data.system_features["stress_displacement"] = stress_displacement
            data.system_features["generator"] = generator
            data.edge_features["vectors"] = vectors
            yield


    def _filter_state_dict(self, state_dict, model):
        """Filter state dict to match model parameters."""
        # Remove keys that don't match the model
        model_keys = set(model.state_dict().keys())
        filtered_dict = {}
        
        for key, value in state_dict.items():
            # Handle potential prefix mismatches (e.g., 'backbone.', 'model.', etc.)
            clean_key = key
            prefixes_to_remove = ['backbone.', 'model.', 'module.']
            for prefix in prefixes_to_remove:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            
            if clean_key in model_keys:
                filtered_dict[clean_key] = value
        
        return filtered_dict



    def sinusoidal_time_embedding_discrete(self, t, dim, T, max_period=10000):
        """
        Sinusoidal embedding for discrete timesteps.

        t: int tensor of shape (...) with values in [0, T-1]
        dim: embedding dimension
        T: total number of discrete diffusion steps
        returns: (..., dim)
        """
        # convert discrete t to float
        t = t.float()

        # optional: normalize to [0, 1]
        # this matches DDPM practice of feeding raw integer t
        t = t / T

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                half, dtype=t.dtype, device=t.device
            ) / half
        )

        args = t[..., None] * freqs[None, :]  # shape (..., half)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb


    def sinusoidal_time_embedding_2(self, t, dim, max_period=10000):
        """
        t: shape (...)
        returns: shape (..., dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=t.dtype, device=t.device) / half
        )
        # t should broadcast properly
        args = t[..., None] * freqs[None, :]  # shape (..., half)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if dim % 2:
            emb = torch.nn.functional.pad(emb, (0,1))

        return emb


    @override
    def model_forward(self, batch, mode: str):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress

        # add the time step here
        t = batch.system_features["t"].float().to(next(self.parameters()).device)
        if self.type == "discrete":
            T = batch.system_features["T"].float().to(next(self.parameters()).device)
            t_emb = self.sinusoidal_time_embedding_discrete(t, 256, T)
        elif self.type == "vp":
            t_emb = self.sinusoidal_time_embedding_2(t, 256)
        else:
            raise ValueError("Invalid type given")
            
        t_emb = t_emb.repeat_interleave(batch.n_node, dim=0)
        batch.system_features["t_emb"] = t_emb

        # Run the backbone
        out = self.backbone(batch)
        node_features = out["node_features"]
        
        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            assert (
                prop := next(
                    (p for p in self.hparams.properties if p.name == name), None
                )
            ) is not None, (
                f"Property {name} not found in properties. "
                "This should not happen, please report this."
            )
            if head is not None:
                res = head(node_features, batch)
                if isinstance(res, torch.Tensor):
                    predicted_properties[name] = res
                elif isinstance(res, dict):
                    if mode!="predict":
                        predicted_properties[name] = res[name]
                    else:
                        predicted_properties.update(res)
                else:
                    raise ValueError(
                        f"Invalid output from head {head}: {res}"
                    )
            else:
                assert isinstance(prop, props.ForcesPropertyConfig) or isinstance(prop, props.StressesPropertyConfig), (
                    f"Conservative Property {name} is not a force or stress property."
                )
                assert "energy" in predicted_properties, ("Energy property is not found for conservative property prediction. Please put energy property before the conservative property in the config.")
                if name in predicted_properties:
                    pass
                else:
                    forces, stress, _ = compute_gradient_forces_and_stress(
                        energy=predicted_properties["energy"],
                        positions=batch.node_features["positions"],
                        displacement=batch.system_features["stress_displacement"],
                        cell=batch.system_features["cell"],
                        training=self.training,
                        compute_stress=self.include_stress,
                        generator=batch.system_features["generator"],
                    )
                    if self.include_forces:
                        predicted_properties["forces"] = forces
                    if self.include_stress:
                        predicted_properties["stresses"] = stress # type: ignore[reportUnboundType]
        
        if "stresses" in predicted_properties and predicted_properties["stress"].shape[1] == 6: # type: ignore[reportUnboundType]
            # Convert the stress tensor to the full 3x3 form
            predicted_properties["stresses"] = voigt_6_to_full_3x3_stress_torch(
                predicted_properties["stresses"] # type: ignore[reportUnboundType]
            )
            
        pred_dict: ModelOutput = {"predicted_properties": predicted_properties}
        return pred_dict
    
    @override
    def model_forward_partition(self, batch, mode: str, using_partition: bool = False):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
        
        # Run the backbone
        out = self.backbone(batch)
        node_features = out["node_features"]
        
        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            assert (
                prop := next(
                    (p for p in self.hparams.properties if p.name == name), None
                )
            ) is not None, (
                f"Property {name} not found in properties. "
                "This should not happen, please report this."
            )
            if head is not None:
                res = head(node_features, batch)
                if isinstance(res, torch.Tensor):
                    predicted_properties[name] = res
                elif isinstance(res, dict):
                    if mode!="predict":
                        predicted_properties[name] = res[name]
                    else:
                        predicted_properties.update(res)
                else:
                    raise ValueError(
                        f"Invalid output from head {head}: {res}"
                    )
            else:
                assert isinstance(prop, props.ForcesPropertyConfig) or isinstance(prop, props.StressesPropertyConfig), (
                    f"Conservative Property {name} is not a force or stress property."
                )
                assert "energy" in predicted_properties, ("Energy property is not found for conservative property prediction. Please put energy property before the conservative property in the config.")
                if name in predicted_properties:
                    pass
                else:
                    forces, stress, _ = compute_gradient_forces_and_stress(
                        energy=predicted_properties["energy"],
                        positions=batch.node_features["positions"],
                        displacement=batch.system_features["stress_displacement"],
                        cell=batch.system_features["cell"],
                        training=self.training,
                        compute_stress=self.include_stress,
                        generator=batch.system_features["generator"],
                    )
                    if self.include_forces:
                        predicted_properties["forces"] = forces
                    if self.include_stress:
                        predicted_properties["stresses"] = stress # type: ignore[reportUnboundType]
        
        if "stresses" in predicted_properties and predicted_properties["stresses"].shape[1] == 6: # type: ignore[reportUnboundType]
            # Convert the stress tensor to the full 3x3 form
            predicted_properties["stresses"] = voigt_6_to_full_3x3_stress_torch(
                predicted_properties["stresses"] # type: ignore[reportUnboundType]
            )
            
        pred_dict: ModelOutput = {"predicted_properties": predicted_properties}
        return pred_dict

    @override
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)

    @override
    def pretrained_backbone_parameters(self):
        return self.backbone.parameters()

    @override
    def output_head_parameters(self):
        for head in self.output_heads.values():
            yield from head.parameters()

    @override
    def cpu_data_transform(self, data):
        return data

    @override
    def collate_fn(self, data_list):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.base import batch_graphs  # type: ignore[reportMissingImports] # noqa

        return batch_graphs(data_list)

    @override
    def gpu_batch_transform(self, batch):
        return batch

    @override
    def batch_to_labels(self, batch):
        # If the labels are not present, throw.
        if not batch.system_targets and not batch.node_targets:
            raise ValueError("No labels found in the batch.")

        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            match prop_type := prop.property_type():
                case "system":
                    assert batch.system_targets is not None, "System targets are None"
                    labels[prop.name] = batch.system_targets[prop.name]
                case "atom":
                    assert batch.node_targets is not None, "Node targets are None"
                    labels[prop.name] = batch.node_targets[prop.name]
                case _:
                    assert_never(prop_type)

        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield import atomic_system  # type: ignore[reportMissingImports] # noqa

        # This is the dataset transform; we can't use GPU here.
        # NOTE: the 0.5.5 version of `orb_models` has a bug in the `ase_atoms_to_atom_graphs`
        # in "orb_models/.../featurization_utilities.py" there is a line: positions = positions.to(device)
        # that trys to move the positions to the device
        # when device="gpu" and num_workers>0, it will throw an error because it is not allowed to do CUDA lazy init in
        # a forked process. We have a patch to fix this, and we have a PR open to fix this upstream. But in 0.5.5 they
        # have not fixed it yet. Until that is merged, a solution is to set device="cpu"
        atom_graphs = atomic_system.ase_atoms_to_atom_graphs(
            atoms,
            system_config=self.system_config,
            device=torch.device("cpu"),
        )
        if "t" in atoms.info:
            atom_graphs.system_features["t"] = torch.tensor([atoms.info["t"]], dtype=torch.float32)
            self.type = atoms.info["type"]
            if atoms.info["type"] == "discrete":
                atom_graphs.system_features["T"] = torch.tensor([atoms.info["T"]], dtype=torch.float32)
 
        if has_labels:
            if atom_graphs.system_targets is None:
                atom_graphs = atom_graphs._replace(system_targets={})

            # Making the type checker happy
            assert atom_graphs.system_targets is not None

            # Also, pass along any other targets/properties. This includes:
            #   - energy: The total energy of the system
            #   - forces: The forces on each atom
            #   - stress: The stress tensor of the system
            #   - anything else you want to predict
            for prop in self.hparams.properties:
                value = prop._from_ase_atoms_to_torch(atoms)
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value.float().numpy())
                    value = torch.from_numpy(value).float().reshape(1, 3, 3)

                match prop_type := prop.property_type():
                    case "system":
                        atom_graphs.system_targets[prop.name] = (
                            value.reshape(1, 1) if value.dim() == 0 else value
                        )
                    case "atom":
                        atom_graphs.node_targets[prop.name] = value # type: ignore[reportUnboundType]
                    case _:
                        assert_never(prop_type)

        # For normalization purposes, we should just pre-compute the composition
        #   vector here and save it in the `system_features`. Then, when batching happens,
        #   we can just use that composition vector from the batched `system_features`.
        atom_types_onehot = F.one_hot(
            atom_graphs.atomic_numbers.view(-1).long(),
            num_classes=120,
        )
        # ^ (n_atoms, 120)
        # Now we need to sum this up to get the composition vector
        composition = atom_types_onehot.sum(dim=0, keepdim=True)
        # ^ (1, 120)
        atom_graphs.system_features["norm_composition"] = composition

        if self.hparams.using_partition and "root_node_indices" in atoms.info:
            root_node_indices = atoms.info["root_node_indices"]
            root_indices_mask = [1 if i in root_node_indices else 0 for i in range(len(atoms))]
            atom_graphs.node_features["root_indices_mask"] = torch.tensor(root_indices_mask, dtype=torch.long)
        
        return atom_graphs
    
    @override
    def get_connectivity_from_data(self, data: AtomGraphs) -> torch.Tensor:
        senders = data.senders.clone()
        receivers = data.receivers.clone()
        return torch.stack([senders, receivers], dim=0)
    
    @override
    def get_connectivity_from_atoms(self, atoms: Atoms) -> np.ndarray:
        twobody_cutoff = self.hparams.system.radius
        edge_indices = neighbor_list_and_relative_vec(
            "vesin",
            pos=np.array(atoms.get_positions()),
            cell=np.array(atoms.get_cell()),
            r_max=twobody_cutoff,
            self_interaction=False,
            pbc=atoms.pbc,
        )
        return edge_indices

    @override
    def create_normalization_context_from_batch(self, batch):
        num_atoms = batch.n_node
        compositions = batch.system_features.get("norm_composition")
        if compositions is None:
            raise ValueError("No composition found in the batch.")
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
    
    @override
    def apply_pruning_message_passing(self, message_passing_steps: int|None):
        """
        Apply message passing for early stopping.
        """
        if message_passing_steps is None:
            pass
        else:
            self.backbone.num_message_passing_steps = min(
                message_passing_steps, self.backbone.num_message_passing_steps
            )


