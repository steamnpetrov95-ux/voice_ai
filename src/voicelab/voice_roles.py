"""Recognisable roles of the actors in the bundled Russian voice library."""

# These describe the actor's credits, not separate character-trained models.
# Match the actor name so existing installations need no model re-download.
ACTOR_ROLES = {
    "владимир зайцев": (
        ("Тони Старк", "«Железный человек»"),
        ("Шерлок Холмс", "фильмы Гая Ричи"),
    ),
    "иван жарков": (
        ("Тор", "киновселенная Marvel"),
        ("Киберслав", "«Киберслав»"),
    ),
    "всеволод кузнецов": (
        ("Геральт", "«Ведьмак 3»"),
        ("Кот в сапогах", "«Шрек 2»"),
    ),
    "андрей ярославцев": (
        ("Оптимус Прайм", "«Трансформеры» (2007)"),
        ("Гимли", "«Властелин колец»"),
    ),
    "никита прозоровский": (
        ("Агент 47", "Hitman: Absolution"),
    ),
    "сергей чихачев": (
        ("Пудж", "Dota 2"),
        ("Earthshaker", "Dota 2"),
    ),
}


def actor_roles(name):
    actor = name.split(" — ")[0].strip().casefold().replace("ё", "е")
    return ACTOR_ROLES.get(actor, ())


def roles_caption(name):
    return " · ".join(role for role, _ in actor_roles(name))


def roles_description(name):
    roles = actor_roles(name)
    if not roles:
        return ""
    return "Озвучивал: " + "; ".join(f"{role} ({work})" for role, work in roles) + "."
