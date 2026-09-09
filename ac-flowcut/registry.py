"""Strategy registry used by the runtime patch."""

STRATEGY_REGISTRY = {}


def register_strategy(name):
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def build_strategy(config):
    try:
        strategy_cls = STRATEGY_REGISTRY[config.method]
    except KeyError as exc:
        raise ValueError(f"No token reduction strategy registered for {config.method}") from exc
    return strategy_cls(config)
