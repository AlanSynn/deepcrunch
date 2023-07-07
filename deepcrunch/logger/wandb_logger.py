from typing import Any, Dict, Optional

import wandb

from deepcrunch.logger.base_logger import BaseLogger
from deepcrunch.utilities.torch_utils import run_on_rank_zero

class WandbLogger(BaseLogger):
    @run_on_rank_zero
    def __init__(
        self,
        proj_name: str,
        model_name: str,
        sys_name: str,
        eval_metric: str,
        threshold: float,
        num_restores: int,
        checkpoint: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self._sys_name = sys_name
        self._eval_metric = eval_metric
        self._threshold = threshold
        self._num_restores = num_restores

        proj = f"{proj_name}_{model_name}"
        run_name = (
            "baseline"
            if num_restores == 0
            else f"{sys_name}_{eval_metric}_{threshold}%_{num_restores}_restores"
        )

        run_id = checkpoint["run_id"] if checkpoint and "run_id" in checkpoint else wandb.util.generate_id()
        assert not checkpoint or (run_name == checkpoint["run_name"] and proj == checkpoint["project_name"])

        self.logger = wandb.init(
            project=proj,
            name=run_name,
            id=run_id,
            resume="allow",
            config=config
        )

    @run_on_rank_zero
    def log(
            self,
            data: Dict[str, Any],
            step: int,
            commit: Optional[bool] = None,
            sync: Optional[bool] = None
        ):
        self.logger.log(data, step, commit, sync)

    @run_on_rank_zero
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "run_id": self.get_attr("id"),
            "project_name": self.get_attr("project"),
            "run_name": self.get_attr("name"),
            "system_name": self._sys_name,
            "eval_metric": self._eval_metric,
            "threshold": self._threshold,
            "num_restores": self._num_restores
        })
        return base_dict