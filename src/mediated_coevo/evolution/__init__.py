__all__ = ["Reflector", "SkillAdvisor"]


def __getattr__(name: str):
    if name == "Reflector":
        from .reflector import Reflector

        return Reflector
    if name == "SkillAdvisor":
        from .skill_advisor import SkillAdvisor

        return SkillAdvisor
    raise AttributeError(name)
