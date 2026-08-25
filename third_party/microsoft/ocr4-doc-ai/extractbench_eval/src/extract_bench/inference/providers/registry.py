from collections.abc import Callable

from extract_bench.inference.providers.base import Provider, ProviderConfigError
from extract_bench.schemas.pipeline import PipelineSpec

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {}


def register_provider(provider_name: str) -> Callable[[type[Provider]], type[Provider]]:
    """
    Decorator to register a Provider class for a given vendor.

    Example:

        @register_provider("llama")
        class LlamaProvider(Provider):
            ...

    Then later:

        provider = create_provider(pipeline_spec)
    """

    def decorator(cls: type[Provider]) -> type[Provider]:
        if provider_name in _PROVIDER_REGISTRY:
            raise ValueError(f"Provider already registered for '{provider_name}'")
        _PROVIDER_REGISTRY[provider_name] = cls
        return cls

    return decorator


def create_provider(pipeline: PipelineSpec) -> Provider:
    """
    Instantiate a Provider for the given PipelineSpec.

    :param pipeline: PipelineSpec with provider, product_type, and config
    :return: Concrete Provider instance
    :raises ProviderConfigError: If provider is not registered
    """
    provider_name = pipeline.provider_name
    provider_cls = _PROVIDER_REGISTRY.get(provider_name)
    if provider_cls is None:
        raise ProviderConfigError(f"No provider registered for '{provider_name}'")

    return provider_cls(
        provider_name=provider_name,
        base_config=pipeline.config,
    )
