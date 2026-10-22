from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceSettings:
    pitch: float = 0.0
    gain: float = 0.0
    room: float = 0.0
    robot: float = 0.0
    radio: bool = False
    chorus: bool = False
    enabled: bool = True
    timbre: float = 0.0
    clarity: float = 2.0
    drive: float = 0.0
    echo: float = 0.0
    quick_pitch: bool = False
    robot_hz: float = 75.0
    flutter: float = 0.0


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    tag: str
    description: str
    symbol: str
    settings: VoiceSettings
    category: str = "Базовые"


PRESETS = (
    Preset("clean", "Чистый голос", "ORIGINAL", "Без дополнительных эффектов, эквализации и компрессии.", "◉", VoiceSettings(clarity=0)),
    Preset("deep", "Низкий голос", "DEEP", "Глубокий тембр для спокойной подачи.", "↓", VoiceSettings(pitch=-4)),
    Preset("bright", "Высокий голос", "BRIGHT", "Лёгкое, высокое звучание.", "↑", VoiceSettings(pitch=4)),
    Preset("robot", "Робот", "SYNTH", "Металлический голос с кольцевой модуляцией.", "⊞", VoiceSettings(robot=0.85)),
    Preset("radio", "Рация", "RADIO", "Узкая полоса частот, как у переговорной рации.", "≋", VoiceSettings(radio=True, gain=3)),
    Preset("monster", "Монстр", "DARK", "Низкий голос с металлическим оттенком.", "◆", VoiceSettings(pitch=-8, robot=0.25, room=0.12)),
    Preset("space", "Космос", "SPACE", "Объёмный голос с хорусом и реверберацией.", "✦", VoiceSettings(pitch=2, room=0.45, chorus=True)),
    Preset("hall", "Большой зал", "REVERB", "Пространственное эхо большого помещения.", "⌂", VoiceSettings(room=0.6)),
)

# Original voice archetypes; these are DSP recipes, not trained character models.
_CHARACTERS = (
    ("aurora", "Аврора", "Женские", 3, -0.25, 0.08, 0, 0, False),
    ("luna", "Луна", "Женские", 2, 0.25, 0.12, 0, 0, False),
    ("mira", "Мира", "Женские", 4, -0.4, 0.02, 0, 0, False),
    ("alina", "Алина", "Женские", 1, -0.2, 0.0, 0, 0, False),
    ("velvet", "Бархат", "Женские", -1, 0.6, 0.1, 0, 0, False),
    ("iskra", "Искра", "Женские", 5, -0.15, 0.04, 0, 0, False),
    ("nord", "Норд", "Мужские", -3, 0.5, 0.0, 0, 0, False),
    ("atlas", "Атлас", "Мужские", -5, 0.7, 0.08, 0, 0, False),
    ("max", "Макс", "Мужские", -2, 0.15, 0.0, 0, 0, False),
    ("narrator", "Диктор", "Мужские", -1, 0.4, 0.04, 0, 0, False),
    ("noir", "Нуар", "Мужские", -4, 0.8, 0.12, 0, 0, False),
    ("pilot", "Пилот", "Мужские", -1, -0.25, 0.0, 0, 0, False),
    ("sakura", "Сакура", "Аниме", 6, -0.35, 0.08, 0, 0, False),
    ("yuki", "Юки", "Аниме", 7, -0.6, 0.04, 0, 0, False),
    ("rin", "Рин", "Аниме", 4, -0.7, 0.0, 0, 0, False),
    ("kaito", "Кайто", "Аниме", -2, -0.15, 0.1, 0, 0, False),
    ("kitsune", "Кицунэ", "Аниме", 5, 0.1, 0.22, 0, 0, True),
    ("neko", "Нэко", "Аниме", 8, -0.45, 0.03, 0, 0, False),
    ("elf", "Эльф", "Фэнтези", 3, -0.1, 0.35, 0, 0, True),
    ("orc", "Орк", "Фэнтези", -7, 0.8, 0.08, 0.12, 3, False),
    ("dwarf", "Гном", "Фэнтези", -4, 1.0, 0.15, 0, 2, False),
    ("mage", "Маг", "Фэнтези", -2, 0.35, 0.45, 0.08, 0, True),
    ("fairy", "Фея", "Фэнтези", 9, -0.5, 0.3, 0, 0, True),
    ("dragon", "Дракон", "Фэнтези", -10, 0.9, 0.3, 0.35, 6, False),
    ("android", "Андроид", "Кибер", 1, -0.3, 0.02, 0.5, 0, False),
    ("cyborg", "Киборг", "Кибер", -3, 0.2, 0.08, 0.65, 3, False),
    ("terminal", "Терминал", "Кибер", -6, -0.6, 0.0, 0.9, 0, False),
    ("hologram", "Голограмма", "Кибер", 4, -0.4, 0.3, 0.3, 0, True),
    ("neon", "Неон", "Кибер", 2, -0.5, 0.15, 0.2, 0, True),
    ("drone", "Дрон", "Кибер", -1, 0.0, 0.2, 1.0, 0, False),
    ("ghost", "Призрак", "Атмосфера", 2, -0.5, 0.65, 0, 0, True),
    ("abyss", "Бездна", "Атмосфера", -9, 1.0, 0.55, 0.15, 0, False),
    ("oracle", "Оракул", "Атмосфера", -1, 0.25, 0.6, 0, 0, True),
    ("dream", "Сон", "Атмосфера", 3, -0.2, 0.5, 0, 0, True),
    ("titan", "Титан", "Игровые", -6, 0.65, 0.12, 0.1, 2, False),
    ("scout", "Разведчик", "Игровые", 1, -0.4, 0.0, 0, 0, False),
    ("boss", "Босс", "Игровые", -5, 0.5, 0.2, 0.2, 4, False),
    ("sprite", "Спрайт", "Игровые", 10, -0.8, 0.1, 0, 0, False),
    ("retro", "Ретро-бот", "Игровые", 3, -0.7, 0.0, 0.7, 2, False),
    ("echo", "Эхо", "Атмосфера", 0, 0.0, 0.2, 0, 0, False),
)

PRESETS += tuple(
    Preset(key, name, category.upper(),
           f"{name}: локальный голосовой образ. Настройте высоту и тембр под свой микрофон.",
           "◉", VoiceSettings(pitch=pitch, timbre=timbre, room=room, robot=robot,
                              drive=drive, chorus=chorus, echo=0.3 if key == "echo" else 0), category)
    for key, name, category, pitch, timbre, room, robot, drive, chorus in _CHARACTERS
)
# Short-delay DSP styles, deliberately distinct from actor/character AI models.
CARTOON_PRESETS = (
    Preset("toon_squirrel", "Бурундучок", "БЫСТРЫЙ", "Звонкий маленький герой: высокий, озорной голос. Мультяшная стилизация.", "↑", VoiceSettings(pitch=9, timbre=-0.6, quick_pitch=True), "Мультяшные"),
    Preset("toon_sidekick", "Маленький помощник", "БЫСТРЫЙ", "Смешной помощник: высокий носовой голос с лёгкой дрожью.", "✦", VoiceSettings(pitch=6, timbre=-0.3, flutter=0.28, quick_pitch=True), "Мультяшные"),
    Preset("toon_giant", "Добрый великан", "БЫСТРЫЙ", "Большой добродушный герой: тёплый бас без металлического эффекта.", "↓", VoiceSettings(pitch=-7, timbre=0.8, quick_pitch=True), "Мультяшные"),
    Preset("toon_goblin", "Гоблин-плут", "БЫСТРЫЙ", "Хриплый хитрец: резкий носовой тембр и лёгкий перегруз.", "◆", VoiceSettings(pitch=3, timbre=-0.7, drive=4, quick_pitch=True), "Мультяшные"),
    Preset("toon_duck", "Утёнок", "БЫСТРЫЙ", "Писклявый, дребезжащий голос мультяшного утёнка.", "↑", VoiceSettings(pitch=7, radio=True, robot=0.35, robot_hz=38, quick_pitch=True), "Мультяшные"),
    Preset("toon_alien", "Пришелец", "БЫСТРЫЙ", "Высокий инопланетный голос с необычными металлическими обертонами.", "✦", VoiceSettings(pitch=4, robot=0.6, robot_hz=115, quick_pitch=True), "Мультяшные"),
    Preset("toon_armor", "Герой в броне", "БЫСТРЫЙ", "Голос через шлем: плотный, с рацией и металлическим оттенком.", "⊞", VoiceSettings(pitch=-2, radio=True, robot=0.22, robot_hz=55, drive=2, quick_pitch=True), "Мультяшные"),
    Preset("toon_villain", "Тёмный страж", "БЫСТРЫЙ", "Мрачный киногерой: низкий шероховатый голос. Не копия конкретного актёра.", "◆", VoiceSettings(pitch=-6, timbre=0.9, drive=5, quick_pitch=True), "Мультяшные"),
    Preset("toon_droid", "Мульт-робот", "БЫСТРЫЙ", "Яркий механический помощник с сильной кольцевой модуляцией.", "⊞", VoiceSettings(robot=0.95, robot_hz=95, radio=True, quick_pitch=True), "Мультяшные"),
    Preset("toon_witch", "Лесная ведьма", "БЫСТРЫЙ", "Скрипучий голос сказочной ведьмы с дрожащей подачей.", "✦", VoiceSettings(pitch=2, timbre=-0.5, flutter=0.65, drive=2, quick_pitch=True), "Мультяшные"),
    Preset("toon_ogre", "Огр", "БЫСТРЫЙ", "Грубый бас большого сказочного существа с рычащими обертонами.", "◆", VoiceSettings(pitch=-10, timbre=1, robot=0.18, robot_hz=32, drive=4, quick_pitch=True), "Мультяшные"),
    Preset("toon_puppet", "Кукольный герой", "БЫСТРЫЙ", "Забавный кукольный персонаж: приподнятый голос с трепещущей окраской.", "◉", VoiceSettings(pitch=5, timbre=0.3, flutter=0.45, quick_pitch=True), "Мультяшные"),
)
PRESETS = (PRESETS[0], *CARTOON_PRESETS, *PRESETS[1:])
PRESET_BY_ID = {preset.id: preset for preset in PRESETS}
CATEGORIES = ("Все голоса", *dict.fromkeys(preset.category for preset in PRESETS))
