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
    hparams.data = MC.AutoSplitDataModuleConfig.draft()
    hparams.data.dataset = MC.XYZDatasetConfig.draft()

    hparams.data.train_split = 0.8
    hparams.data.dataset.T = args_dict["T"]
    hparams.data.dataset.sigma_min = args_dict["sigma_min"]
    hparams.data.dataset.sigma_max= args_dict["sigma_max"]
    hparams.data.dataset.src = "./data/relaxed.xyz"

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

    # Configure Model Checkpoint
    ckpt_name = f"{args_dict['model_type']}-best"
    if os.path.exists(f"./checkpoints/{ckpt_name}.ckpt"):
        os.remove(f"./checkpoints/{ckpt_name}.ckpt")
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        dirpath="./checkpoints",
        filename=ckpt_name,
        save_top_k=1,
        mode="min",
        every_n_epochs=100,
    )

    #Configure Logger
    hparams.trainer.loggers = [
        WandbLoggerConfig(
            project="Diffusion-Pretraining-Orb_v2-new",
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
    "batch_size" : 32,
    "devices" : [0],
    "max_epochs" : 200,
    "T": 32,
    "sigma_min": 0.05,
    "sigma_max": 1
}

mt_config = hparams(args_dict)
model, trainer = MatterTuner(mt_config).tune()
trainer.save_checkpoint(f"./cpkts/diffusion_T-{args_dict["T"]}_sigmamin-{args_dict["sigma_min"]}_sigmamax-{args_dict["sigma_max"]}_val.cpkt")

