"""Cases ported from Neleceni's test/match.test.ts (real OCR readings), adapted to MP ids."""

from reader.match import OcrLine, build_index, classify, line_matches_name, normalize, pick_speaker_line, similar_lines

MPS = build_index([
    {"id": 1050, "l": "Milan Radin", "c": "Милан Радин"},
    {"id": 30, "l": "Aleksandar Vučić", "c": "Александар Вучић"},
    {"id": 1029, "l": "Milenko Jovanov", "c": "Миленко Јованов"},
    {"id": 1190, "l": "Ana Brnabić", "c": "Ана Брнабић"},
    {"id": 59, "l": "Nataša Jovanović", "c": "Наташа Јовановић"},
    {"id": 1549, "l": "Milica Jovanović", "c": "Милица Јовановић"},
    {"id": 1587, "l": "Vukašin Vučićević", "c": "Вукашин Вучићевић"},
    {"id": 1602, "l": "Dina Vučinić", "c": "Дина Вучинић"},
])
V = "Александар Вучић"


def line(text, y0, conf=80, x0=0):
    return OcrLine(text, conf, x0, y0, x0 + 600, y0 + 40)


def kind(lines):
    o = classify(lines, MPS, 35)
    return o.kind, o.mp


def test_normalize():
    assert normalize("Александар Вучић") == "aleksandar vucic"
    assert normalize("Aleksandar Vučić") == "aleksandar vucic"
    assert normalize("ЂОРЂЕ Џаковић, Љиљана") == "djordje dzakovic ljiljana"


def test_line_matches_name():
    for ok in ["Александар Вучић", "__Александар Вучић, | а", "__Александар Вучи ___ а", "АлександарВучић",
               "ALEKSANDAR VUČIĆ", "А екса · р Вучић »“' =“", "ЦМ екса Вучић то", "АпењСандар Вучић", "м аксанћр Вучић ид"]:
        assert line_matches_name(ok, V), ok
    for bad in ["Ана Вучић", "Александра Вуковић", "Александар Ђурић", "екса уци", "Ана Брнабић",
                "Александар Јовановић", "Милош Вучевић", "Vuk Vučković", "Александар Вулин", "Андрија Вучић"]:
        assert not line_matches_name(bad, V), bad


def test_pick_speaker_line():
    assert pick_speaker_line([line("– Александар Вучић - Србија побеђује", 85), line("Ана Брнабић", 40)]).text == "Ана Брнабић"
    assert pick_speaker_line([line("Ммм Шшш", 5, 40), line("Миленко Јованов", 110), line("Александар Вучић - Србија побеђује", 152)]).text == "Миленко Јованов"
    assert pick_speaker_line([line("Александар Вучић Србија побеђује", 85)]) is None
    assert pick_speaker_line([line("Ана Брнабић", 0), line("Александар Вучић - Србија побеђује", 120)]) is None
    assert pick_speaker_line([line("Ана Брнабић", 60), line("Председник Народне скупштине", 104)]).text == "Ана Брнабић"


def test_group_line_never_decides():
    # an SNS deputy: Vučić is only in the group line
    assert kind([line("Ана Брнабић", 18), line("Александар Вучић - Србија побеђује", 82)]) == ("mp", 1190)
    assert kind([line("Александар Вучић - Србија побеђује", 82)]) == ("none", None)
    assert kind([line("Александар Вучић Србија", 152)]) == ("none", None)


def test_real_replay_readings():
    r = lambda top, bottom, c=50: [line(top, 18, c), line(bottom, 82, c)]
    assert kind(r("Миленко Јованов —", "Александар ВучиЋ- УСрбија побеђује", 54)) == ("mp", 1029)
    assert kind(r("Миленко Јованбв. “У __", "Александар ВучтЋ УСрбија побеђује.", 41)) == ("mp", 1029)
    assert kind(r("Александар Вучић :.", "Александар Вучић обубија побеђује", 92)) == ("mp", 30)
    assert kind(r('"Халексан р Вучић “> 47 =“ ние', "жрксан о Вужј у Србија побеђује и", 38)) == ("mp", 30)
    assert kind(r('"Фалекса дар Вучић % =', "фудександар Ву у Србија побеђује |")) == ("mp", 30)
    assert kind([line("__Алекбиндарвучнијрдајр Вучић - Сромја п", 18, 57)])[0] == "none"


def test_split_and_junk_lines():
    lines = [line("У ексанћр Вучић", 110, 51, 40), line("ко", 112, 25, 700), line(",ександар Ву ~", 152, 62, 30),
             line("у", 154, 71, 400), line(". у Србија побеђује", 151, 71, 560)]
    assert kind(lines) == ("mp", 30)
    lines = [line("о ЖАО г,", 10, 0, 20), line("м ексанр Вучић “и", 100, 56, 40), line("« ександар Ву ‘“ ..уСрбија побеђује", 150, 59, 30)]
    assert kind(lines) == ("mp", 30)


def test_same_surname_needs_first_name():
    assert kind([line("Милица Јовановић", 60)]) == ("mp", 1549)
    assert kind([line("Наташа Јовановић", 60)]) == ("mp", 59)
    assert kind([line("Вукашин Вучићевић", 60)]) == ("mp", 1587)


def test_other_speakers():
    assert kind([line("Милош Вучевић", 60, 80)]) == ("other", None)  # not an MP: minister
    assert kind([line("Аракоашир Учи Вав", 18, 28)]) == ("none", None)  # not confident
    assert kind([line("ииии тттт сссс", 30, 70)]) == ("none", None)  # lower-case noise
    # a garbled reading of the current speaker is not "someone else"
    assert classify([line("Хиоса ћр ву Вучић", 60, 63)], MPS, 35, current=30).kind == "none"


def test_similar_lines():
    assert similar_lines("Александар Вучић", "Александар Вучи")
    assert not similar_lines("Александар Вучић", "Ана Брнабић")


def test_garbled_current_speaker_is_not_someone_else():
    # live 24 Nov 2026: "Милан Рајан" read three times split Radin's speech in two
    assert classify([line("Милан Рајан", 60, 70)], MPS, 35, current=1050).kind == "none"
    assert classify([line("Милан Рајан", 60, 70)], MPS, 35).kind == "other"  # without context it's just a name
    assert classify([line("Милош Вучевић", 60, 80)], MPS, 35, current=1050).kind == "other"  # a real change still counts
