import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.nn as nn

from olmo_core.nn.transformer import Transformer

from ..config import Config, Registrable
from ..exceptions import OLMoConfigurationError
from ..utils import get_default_device, move_to_device
from .mup import MuPConfig, mup_lr_multipliers, split_groups_by_lr_multiplier

__all__ = [
    "OptimConfig",
    "OptimGroupOverride",
]

log = logging.getLogger(__name__)

Opt = TypeVar("Opt", bound=torch.optim.Optimizer)

LR_FIELD = "lr"
INITIAL_LR_FIELD = "initial_lr"


@dataclass
class OptimGroupOverride(Config):
    params: List[str]
    """
    A list of fully qualified parameter names (FQNs) or wild card to match FQNs.
    """

    opts: Dict[str, Any]
    """
    Options to set in the corresponding param group.
    """


@dataclass
class OptimConfig(Config, Registrable, Generic[Opt], metaclass=ABCMeta):
    """
    Base class for :class:`~torch.optim.Optimizer` configs.
    """

    group_overrides: Optional[List[OptimGroupOverride]] = None
    """
    Use this to pull out groups parameters into a separate param groups with their own options.
    """

    compile: bool = False
    """
    Compile the optimizer step.

    .. warning::
        Optimizer step compilation is still in beta and may not work with some optimizers.
        You could also see unexpected behavior and very poor performance when turning this feature
        on in the middle of a run that was previously trained without compiling the optimizer
        due to the LR being restored to a float instead of a tensor.
    """

    fixed_fields: Tuple[str, ...] = (INITIAL_LR_FIELD,)
    """
    These are fields that should not be overridden by the value in a checkpoint after
    loading optimizer state.
    """

    mup: Optional[MuPConfig] = None
    """
    Scale each parameter's learning rate following muP. Every parameter group, including the
    ones from :data:`group_overrides`, is split by learning-rate multiplier.
    """

    @property
    def device(self) -> torch.device:
        return get_default_device()

    def _apply_mup(
        self, group_overrides: List[OptimGroupOverride], model: nn.Module
    ) -> List[OptimGroupOverride]:
        if self.mup is None:
            return group_overrides
        if not hasattr(self, LR_FIELD):
            raise OLMoConfigurationError(f"muP requires {self.__class__.__name__} to have an 'lr'")
        return split_groups_by_lr_multiplier(
            group_overrides,
            mup_lr_multipliers(model, self.mup),
            self.mup,
            lr=getattr(self, LR_FIELD),
            weight_decay=getattr(self, "weight_decay", None),
        )

    def _expand_param_globs(
        self,
        go: OptimGroupOverride,
        all_params: Dict[str, Any],
        frozen_param_names: Set[str],
        g_idx: int,
        strict: bool = True,
    ) -> OptimGroupOverride:
        param_names: List[str] = []
        for pattern in go.params:
            matches = 0
            for name in list(all_params.keys()):
                if fnmatch(name, pattern):
                    param_names.append(name)
                    matches += 1

            if matches == 0:
                for name in frozen_param_names:
                    if fnmatch(name, pattern):
                        log.warning(
                            f"optim group {g_idx} override pattern '{pattern}' matches a frozen parameter and will be ignored"
                        )
                        break
                else:
                    msg = f"optim group {g_idx} override pattern '{pattern}' does not match any parameters"
                    if strict:
                        raise OLMoConfigurationError(msg)
                    else:
                        log.warning(msg)

        return OptimGroupOverride(param_names, go.opts.copy())

    def build_groups(
        self, model: nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: Dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                all_params[n] = p
            else:
                frozen_params.add(n)

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overridden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overridden_param_names], {}
        )
        group_overrides.append(default_override)
        group_overrides = self._apply_mup(group_overrides, model)

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]

    @classmethod
    @abstractmethod
    def optimizer(cls) -> Type[Opt]:
        """
        Get the optimizer class associated with this config.
        """
        raise NotImplementedError

    def build(self, model: nn.Module, strict: bool = True) -> Opt:
        """
        Build the optimizer. This default implementation is suitable for standard, point-wise
        optimizers such as AdamW, Lion, etc.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")
        kwargs.pop("mup")

        optim: torch.optim.Optimizer = self.optimizer()(
            self.build_groups(model, strict=strict), **kwargs
        )

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: Optional[float] = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Opt, optim)


@dataclass
class MatrixAwareOptimConfig(OptimConfig, Generic[Opt]):
    """
    Configuration class for building a matrix-aware optimizer.
    """

    @classmethod
    def optimizer(cls) -> Type[Opt]:
        raise NotImplementedError

    def categorize_parameters(self, model: nn.Module) -> Dict[str, List[str]]:
        assert isinstance(model, Transformer)

        embed_params = [
            f"embeddings.{n}" for n, p in model.embeddings.named_parameters() if p.ndim == 2
        ]
        matrix_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim == 2]
        vector_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim < 2]
        vector_params += [f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim < 2]
        lm_head_params = [
            f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim == 2
        ]

        # Convolution layers typically use parameter tensors with 3+ dimensions. These are currently
        # not supported. Experimental support for for 3+ dim parameters is available in the Muon
        # optimizer. It simply flattens the parameters into a 2D tensor.
        # Check for 3D+ parameters which are not supported
        params_3d_plus = [n for n, p in model.named_parameters() if p.ndim > 2]
        assert not params_3d_plus, f"3D+ parameters are not supported: {params_3d_plus}"

        # Assert all parameters are categorized
        all_model_params = {n for n, p in model.named_parameters() if p.requires_grad}
        categorized_params = set(embed_params + matrix_params + vector_params + lm_head_params)
        uncategorized = all_model_params - categorized_params
        assert not uncategorized, f"Uncategorized parameters: {uncategorized}"

        # Assert no params are in multiple categories
        all_categorized = embed_params + matrix_params + vector_params + lm_head_params
        assert len(all_categorized) == len(
            set(all_categorized)
        ), "Some parameters are in multiple categories"

        return {
            "embed": embed_params,
            "matrix": matrix_params,
            "vector": vector_params,
            "lm_head": lm_head_params,
        }

    def default_group_overrides(self, model: nn.Module) -> List[OptimGroupOverride]:
        """
        Default group overrides for matrix-aware optimizers.
        """
        raise NotImplementedError

    def build_groups(
        self, model: torch.nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                all_params[n] = p
            else:
                frozen_params.add(n)

        if self.group_overrides is not None:
            raise RuntimeError(
                "Manual group_overrides are not currently supported for Matrix-Aware optimizers"
            )
        self.group_overrides = self.default_group_overrides(model)

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overridden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overridden_param_names], {}
        )
        group_overrides.append(default_override)

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]

    def create_optimizer(self, model: torch.nn.Module, strict: bool = True) -> Opt:
        """
        Create the optimizer.
        """
        raise NotImplementedError

    def build(self, model: torch.nn.Module, strict: bool = True) -> Opt:
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        if self.mup is not None:
            raise OLMoConfigurationError(
                f"muP learning rates are not supported for {self.__class__.__name__}"
            )

        kwargs = self.as_dict(exclude_private_fields=True)
        kwargs.pop("mup")
        group_overrides = kwargs.pop("group_overrides")
        if group_overrides is not None:
            log.warning(f"group_overrides is set but will be ignored for {self.__class__.__name__}")
        compile_opt = kwargs.pop("compile")
        if compile_opt:
            log.warning(f"compile is set to True but will be ignored for {self.__class__.__name__}")
        kwargs.pop("fixed_fields")

        optim = self.create_optimizer(model, strict=strict, **kwargs)

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: list[dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: float | None = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            raise NotImplementedError(
                "Compiling optimizer step is not supported for Matrix-Aware optimizers"
            )

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Opt, optim)
