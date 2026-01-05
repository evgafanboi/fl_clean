from dataclasses import dataclass
from typing import Any, Callable, Dict

from .strategy.FedAvg import FedAvg
from .strategy.FedCoMed import FedCoMed
from .strategy.FedDyn import FedDyn
from .strategy.FedProx import FedProx
from .strategy.robust_filter import RobustFilterWeights
from .strategy.DeepFed import DeepFed


@dataclass
class StrategyRuntime:
    name: str
    aggregator: Any
    client_strategy: Any
    requires_participant_ids: bool = False


@dataclass
class StrategyConfig:
    aggregator_factory: Callable[[Dict[str, Any]], Any]
    client_factory: Callable[[Any, Dict[str, Any]], Any]
    requires_participant_ids: bool = False

    def build(self, params: Dict[str, Any]) -> StrategyRuntime:
        aggregator = self.aggregator_factory(params)
        client_strategy = self.client_factory(aggregator, params)
        name = getattr(aggregator, 'name', params.get('strategy_name', 'Strategy'))
        return StrategyRuntime(name=name, aggregator=aggregator, client_strategy=client_strategy, requires_participant_ids=self.requires_participant_ids)


def _fedavg_aggregator_factory(_: Dict[str, Any]) -> FedAvg:
    return FedAvg()


def _fedavg_client_factory(aggregator: FedAvg, _: Dict[str, Any]) -> FedAvg:
    return aggregator


def _fedprox_factory(params: Dict[str, Any]) -> FedProx:
    mu = params.get('mu', 0.01)
    adaptive_mu = params.get('adaptive_mu', False)
    return FedProx(mu=mu, adaptive_mu=adaptive_mu)


def _fedprox_client_factory(aggregator: FedProx, _: Dict[str, Any]) -> FedProx:
    return aggregator


def _feddyn_factory(params: Dict[str, Any]) -> FedDyn:
    alpha = params.get('feddyn_alpha', params.get('alpha', 0.1))
    return FedDyn(alpha=alpha)


def _fedcomed_aggregator_factory(_: Dict[str, Any]) -> FedCoMed:
    return FedCoMed()


def _fedcomed_client_factory(_: Any, __: Dict[str, Any]) -> FedAvg:
    return FedAvg()


def _robust_filter_factory(params: Dict[str, Any]) -> RobustFilterWeights:
    epsilon = params.get('epsilon', params.get('robust_epsilon', 0.2))
    tau = params.get('robust_tau', 0.1)
    return RobustFilterWeights(epsilon=epsilon, tau=tau)


def _robust_filter_client_factory(_: Any, __: Dict[str, Any]) -> FedAvg:
    return FedAvg()


def _deepfed_aggregator_factory(params: Dict[str, Any]) -> DeepFed:
    key_length = params.get('key_length', 1024)  # key length
    return DeepFed(key_length=key_length)


def _deepfed_client_factory(aggregator: DeepFed, _: Dict[str, Any]) -> DeepFed:
    return aggregator


STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {
    'FedAvg': StrategyConfig(
        aggregator_factory=_fedavg_aggregator_factory,
        client_factory=_fedavg_client_factory
    ),
    'FedProx': StrategyConfig(
        aggregator_factory=_fedprox_factory,
        client_factory=_fedprox_client_factory
    ),
    'FedDyn': StrategyConfig(
        aggregator_factory=_feddyn_factory,
        client_factory=lambda aggregator, params: aggregator,
        requires_participant_ids=True
    ),
    'FedCoMed': StrategyConfig(
        aggregator_factory=_fedcomed_aggregator_factory,
        client_factory=_fedcomed_client_factory
    ),
    'RobustFilter': StrategyConfig(
        aggregator_factory=_robust_filter_factory,
        client_factory=_robust_filter_client_factory
    ),
    'DeepFed': StrategyConfig(
        aggregator_factory=_deepfed_aggregator_factory,
        client_factory=_deepfed_client_factory
    )
}


def build_strategy(strategy_name: str, params: Dict[str, Any]) -> StrategyRuntime:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {', '.join(sorted(STRATEGY_REGISTRY))}")

    params = dict(params)
    params['strategy_name'] = strategy_name
    return STRATEGY_REGISTRY[strategy_name].build(params)
