from templater.base import BaseTemplater
from templater.physiq_verified import PVideoTemplater, SoraTemplater

REGISTRY: dict[str, type[BaseTemplater]] = {
    "base": BaseTemplater,
    "pvideo": PVideoTemplater,
    "sora": SoraTemplater,
}
