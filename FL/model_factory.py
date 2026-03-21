from typing import Callable, Dict, Optional

from models import dense, gru, dcblstm, fedprox_wrapper, feddyn_wrapper, fedmlb_wrapper

from .aggregators import StrategyRuntime


ModelBuilder = Callable[[int, int, int, StrategyRuntime, Optional[int]], object]


def _dense_builder(
    input_dim: int,
    num_classes: int,
    batch_size: int,
    strategy_runtime: StrategyRuntime,
    client_id: Optional[int]
):
    strategy_name = getattr(strategy_runtime.client_strategy, 'name', '').lower()

    if strategy_name == 'fedora':
        return dense.create_dense_model(input_dim, num_classes, batch_size)
    if strategy_name == 'fedprox':
        return fedprox_wrapper.create_fedprox_dense_model(input_dim, num_classes, batch_size, strategy_runtime.client_strategy)
    if strategy_name == 'feddyn':
        return feddyn_wrapper.create_feddyn_dense_model(input_dim, num_classes, batch_size, strategy_runtime.client_strategy, client_id)
    if strategy_name == 'fedmlb':
        return fedmlb_wrapper.create_fedmlb_model(
            input_dim,
            num_classes,
            batch_size,
            lambda1=strategy_runtime.client_strategy.lambda1,
            lambda2=strategy_runtime.client_strategy.lambda2,
            temperature=strategy_runtime.client_strategy.temperature
        )
    return dense.create_dense_model(input_dim, num_classes, batch_size)


def _gru_builder(
    input_dim: int,
    num_classes: int,
    batch_size: int,
    strategy_runtime: StrategyRuntime,
    client_id: Optional[int]
):
    strategy_name = getattr(strategy_runtime.client_strategy, 'name', '').lower()

    skip_compile = strategy_name == 'feddyn'
    base = gru.create_gru_model(input_dim, num_classes, batch_size, compile=not skip_compile)
    if strategy_name == 'fedprox':
        return fedprox_wrapper.FedProxModelWrapper(base, strategy_runtime.client_strategy)
    if strategy_name == 'feddyn':
        return feddyn_wrapper.FedDynModelWrapper(base, strategy_runtime.client_strategy, client_id)
    return base


def _dcblstm_builder(
    input_dim: int,
    num_classes: int,
    batch_size: int,
    strategy_runtime: StrategyRuntime,
    client_id: Optional[int],
):
    strategy_name = getattr(strategy_runtime.client_strategy, 'name', '').lower()

    skip_compile = strategy_name == 'feddyn'
    base = dcblstm.create_dcblstm_model(input_dim, num_classes, batch_size, compile=not skip_compile)
    if strategy_name == 'fedprox':
        return fedprox_wrapper.FedProxModelWrapper(base, strategy_runtime.client_strategy)
    if strategy_name == 'feddyn':
        return feddyn_wrapper.FedDynModelWrapper(base, strategy_runtime.client_strategy, client_id)
    return base

MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    'dense': _dense_builder,
    'gru': _gru_builder,
    'dcblstm': _dcblstm_builder,
}


def create_model(
    architecture: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    strategy_runtime: StrategyRuntime,
    client_id: Optional[int] = None
):
    architecture_key = architecture.lower()
    if architecture_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture '{architecture}'. Available: {', '.join(sorted(MODEL_REGISTRY))}")
    builder = MODEL_REGISTRY[architecture_key]
    return builder(input_dim, num_classes, batch_size, strategy_runtime, client_id)
