import mattertune as mt
from mattertune import MatterTuner
import mattertune.configs as MC
import os
from pathlib import Path
import ase
from mattertune.backbones import (
    EqV2BackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
)
from mattertune.configs import WandbLoggerConfig


def hparams(args_dict):
    hparams = MC.MatterTunerConfig.draft()

    hparams.model = MC.ORBBackboneConfig.draft()
    hparams.model.pretrained_model = args_dict["model_type"]
    if args_dict['ckpt_path'] is not None:
        hparams.model.checkpoint_path = args_dict['ckpt_path']
    

    hparams.model.ignore_gpu_batch_transform_error = True
    hparams.model.freeze_backbone = False
    hparams.model.reset_output_heads = True
    hparams.model.optimizer = MC.AdamWConfig(
        lr=5e-5,
        weight_decay=0,
    )

    # Add model properties
    hparams.model.properties = []
    property = MC.NoisePropertyConfig(
        loss = MC.MSELossConfig(),
        dtype= "float",
        name="noise",
    )

    hparams.model.properties.append(property)

    ## Data Hyperparameters
    hparams.data = MC.ManualSplitDataModuleConfig.draft()
    hparams.data.train = MC.XYZDatasetConfig.draft()

    hparams.data.train.T = args_dict["T"]
    hparams.data.train.sigma_min = args_dict["sigma_min"]
    hparams.data.train.sigma_max= args_dict["sigma_max"]
    hparams.data.train.diffusion_type = "vp"

    hparams.data.train.src =  args_dict["src"]
    
    hparams.data.batch_size = args_dict["batch_size"]
    hparams.data.num_workers = 0
    hparams.data.pin_memory = False

    ## Trainer Hyperparameters
    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = args_dict["max_epochs"]
    hparams.trainer.accelerator = "gpu"
    hparams.trainer.devices = args_dict["devices"]
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = 1.0
    hparams.trainer.precision = "32"

    name = args_dict["name"]

    # Configure Model Checkpoint
    ckpt_name = f"{args_dict['model_type']}-best-VP-.05-1_{name}"
    if os.path.exists(f"./checkpoints/{ckpt_name}.ckpt"):
        os.remove(f"./checkpoints/{ckpt_name}.ckpt")
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        dirpath="./checkpoints",
        filename=ckpt_name,
        save_top_k=1,
        mode="min",
        every_n_epochs=50,
    )

    # Configure Logger
    hparams.trainer.loggers = [
        WandbLoggerConfig(
            project="Diffusion-Pretraining-Orb-v2-VP",
            name=f"Pretrain-diffusion-{args_dict['model_type']}",
        )
    ]

    # Additional trainer settings
    hparams.trainer.additional_trainer_kwargs = {
        "inference_mode": False,
    }

    hparams = hparams.finalize(strict=False)
    return hparams

args_dict = {
    "model_type" : "orb-v2",
    "batch_size" : 16,
    "devices" : [0],
    "max_epochs" : 400,
    "T": 32,
    "sigma_min": 0.1,
    "sigma_max": 2,
    "src":"/storage/aamod/projects/data/final/20k_synth.xyz",
    "ckpt_path" : None,
    "name" : "20k_synth"
}

mt_config = hparams(args_dict)
model, trainer = MatterTuner(mt_config).tune()
filename = (
    f"diffusion_VP-synth-_T-{args_dict['T']}_"
    f"{args_dict['sigma_min']}-{args_dict['sigma_max']}-"
    f"{args_dict['max_epochs']}_{args_dict['name']}.cpkt"
)

trainer.save_checkpoint(f"./cpkts/{filename}")

