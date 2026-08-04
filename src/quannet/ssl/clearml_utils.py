"""Optional ClearML integration for SSL pretrain/fine-tune runs.

Every hook here is a no-op unless the caller actually passes --clearml-project
(or, for resolve_ssl_checkpoint, --clearml-ssl-checkpoint) - quannet.ssl.pretrain
and scripts/finetune_ssl_resnet.py keep working standalone (e.g. the CPU
debug runs documented in docker/ssl/README.md) without a ClearML server
configured. `clearml` itself is only imported inside these functions so the
rest of the SSL package has no hard dependency on it.
"""
from typing import List, Optional
from pathlib import Path

import lightning as L


def add_clearml_args(parser):
    group = parser.add_argument_group('ClearML (all optional - omit to run without tracking)')
    group.add_argument(
        '--clearml-project', default=None, help='enables ClearML tracking; the project to log this run under'
    )
    group.add_argument('--clearml-task-name', default=None, help="defaults to --output-dir's basename")
    group.add_argument(
        '--clearml-tags',
        nargs='*',
        default=None,
        help='tags attached to the ClearML task',
    )
    group.add_argument(
        '--clearml-dataset',
        action='append',
        default=[],
        metavar='DATASET_PROJECT/DATASET_NAME|DATASET_ID',
        help='ClearML Dataset to pull as an ESP-cache root (repeatable). Its local copy is '
        'appended to --data-dirs / used as --cache_root.',
    )
    return parser


def init_task(project: Optional[str], task_name: str, tags=None):
    """Returns a clearml.Task, or None if tracking wasn't requested (no --clearml-project)."""
    if not project:
        return None
    from clearml import Task

    return Task.init(project_name=project, task_name=task_name, tags=tags)


def resolve_data_dirs(data_dirs: List[str], clearml_dataset_refs: List[str]) -> List[str]:
    """Appends the local copy of every --clearml-dataset ref to data_dirs."""
    if not clearml_dataset_refs:
        return list(data_dirs)
    from clearml import Dataset

    resolved = list(data_dirs)
    for ref in clearml_dataset_refs:
        if '/' in ref:
            dataset_project, dataset_name = ref.split('/', 1)
            ds = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name)
        else:
            ds = Dataset.get(dataset_id=ref)
        resolved.append(ds.get_local_copy())
    return resolved


def resolve_ssl_checkpoint(local_path: Optional[Path], clearml_task_ref: Optional[str]) -> Optional[Path]:
    """--clearml-ssl-checkpoint takes precedence: pulls the last OutputModel of that pretrain Task."""
    if not clearml_task_ref:
        return local_path
    from clearml import Task

    task = Task.get_task(task_id=clearml_task_ref)
    output_models = task.models['output']
    if not output_models:
        raise ValueError(f'ClearML task {clearml_task_ref} has no output models to fine-tune from')
    return Path(output_models[-1].get_local_copy())


def upload_output_model(task, path: Path, name: str):
    if task is None:
        return
    from clearml import OutputModel

    out_model = OutputModel(task=task, name=name, framework='PyTorch')
    out_model.update_weights(weights_filename=str(path))


class ClearMLCheckpointUpload(L.Callback):
    """Uploads the last Lightning checkpoint to the ClearML task every N epochs.

    Belt-and-suspenders against preemptible/spot GPU instances: local
    Lightning checkpoints already resume via --resume-from-checkpoint, but if
    the instance is torn down (disk included), this keeps the most recent
    epochs' progress recoverable from the ClearML task instead of losing the
    whole run.
    """

    def __init__(self, task, checkpoint_dir: Path, every_n_epochs: int = 1):
        self.task = task
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_epochs = max(1, every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.task is None or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        last_ckpt = self.checkpoint_dir / 'last.ckpt'
        if last_ckpt.exists():
            self.task.upload_artifact(name='last_checkpoint', artifact_object=str(last_ckpt))
