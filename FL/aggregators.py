from dataclasses import dataclass
from typing import Any, Callable, Dict

from .strategy.FedAvg import FedAvg
from .strategy.FedCoMed import FedCoMed
from .strategy.FedDyn import FedDyn
from .strategy.FedProx import FedProx
from .strategy.robust_filter import RobustFilterWeights
from .strategy.DeepFed import DeepFed
from .strategy.FLTrust import FLTrust
from .strategy.FedMLB import FedMLB
from .strategy.NoneStrategy import NoneStrategy
from .strategy.SecureAggregation import SecureAggregation
from .strategy.FedSSDexp import FedSSDexp


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
    return FedProx(mu=mu)


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
    return RobustFilterWeights(epsilon=epsilon)


def _robust_filter_client_factory(_: Any, __: Dict[str, Any]) -> FedAvg:
    return FedAvg()


def _deepfed_aggregator_factory(params: Dict[str, Any]) -> DeepFed:
    key_length = params.get('key_length', 1024)  # key length
    return DeepFed(key_length=key_length)


def _deepfed_client_factory(aggregator: DeepFed, _: Dict[str, Any]) -> DeepFed:
    return aggregator


def _fltrust_factory(params: Dict[str, Any]) -> FLTrust:
    root_iterations = params.get('root_iterations', 1)
    return FLTrust(root_iterations=root_iterations)


def _fltrust_client_factory(aggregator: FLTrust, _: Dict[str, Any]) -> FLTrust:
    return aggregator


def _fedmlb_factory(params: Dict[str, Any]) -> FedMLB:
    lambda1 = params.get('lambda1', 1.0)
    lambda2 = params.get('lambda2', 1.0)
    temperature = params.get('temperature', 1.0)
    return FedMLB(lambda1=lambda1, lambda2=lambda2, temperature=temperature)


def _fedmlb_client_factory(aggregator: FedMLB, _: Dict[str, Any]) -> FedMLB:
    return aggregator


def _none_factory(params: Dict[str, Any]) -> NoneStrategy:
    return NoneStrategy()


def _none_client_factory(aggregator: NoneStrategy, _: Dict[str, Any]) -> NoneStrategy:
    return aggregator


def _secagg_factory(_: Dict[str, Any]) -> SecureAggregation:
    return SecureAggregation()


def _secagg_client_factory(aggregator: SecureAggregation, _: Dict[str, Any]) -> SecureAggregation:
    return aggregator



def _fedssdexp_factory(params: Dict[str, Any]) -> FedSSDexp:
    m_max = params.get('m_max', 1.0)
    use_support = params.get('support', False)
    return FedSSDexp(m_max, use_support)


def _fedssdexp_client_factory(aggregator: FedSSDexp, _: Dict[str, Any]) -> FedSSDexp:
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
    ),
    'FLTrust': StrategyConfig(
        aggregator_factory=_fltrust_factory,
        client_factory=_fltrust_client_factory
    ),
    'FedMLB': StrategyConfig(
        aggregator_factory=_fedmlb_factory,
        client_factory=_fedmlb_client_factory
    ),
    'SecureAggregation': StrategyConfig(
        aggregator_factory=_secagg_factory,
        client_factory=_secagg_client_factory
    ),
    'FedSSDexp': StrategyConfig(
        aggregator_factory=_fedssdexp_factory,
        client_factory=_fedssdexp_client_factory
    ),
    'None': StrategyConfig(
        aggregator_factory=_none_factory,
        client_factory=_none_client_factory
    )
}


def build_strategy(strategy_name: str, params: Dict[str, Any]) -> StrategyRuntime:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {', '.join(sorted(STRATEGY_REGISTRY))}")

    params = dict(params)
    params['strategy_name'] = strategy_name
    return STRATEGY_REGISTRY[strategy_name].build(params)
