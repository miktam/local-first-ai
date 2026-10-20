"""exp_045 voice path: what the voice reads, how spoken phone numbers are joined, and the noise level."""
import numpy as np

from scripts.notetaker import join_spoken_numbers
from scripts.voice45 import SNR_DB, pink_noise, speakable


def test_speakable_drops_emoji_and_keeps_text():
    assert speakable("Hola 👋🏻 busco piso ❤️ en Marbella, 300.000 €") == "Hola busco piso en Marbella, 300.000 €"


def test_spoken_phone_groups_are_joined():
    assert join_spoken_numbers("joindre au plus 44, 77, 00, 90, 09, 01 pour") == "joindre au +44 77 00 90 09 01 pour"
    assert join_spoken_numbers("Связь, плюс 44, 77, 00, 90, 01, 54, Рустам.") == "Связь, +44 77 00 90 01 54, Рустам."
    assert join_spoken_numbers("unter plus 44, 7700, 900103.") == "unter +44 7700 900103."
    assert join_spoken_numbers("llámame al 06, 12, 34, 56, 78") == "llámame al 06 12 34 56 78"     # national, leading 0


def test_prices_dates_and_short_runs_are_left_alone():
    for s in ("12, 00, 15, 00 par mois", "Preisvorstellung 295.000, 300.000 oder", "in 2026, 2027, 2028 we come",
              "around 700 to 800,000", "Budget bis 1, 4 Millionen"):
        assert join_spoken_numbers(s) == s


def test_pink_noise_is_unit_rms_and_seeded():
    a, b = pink_noise(16000, 7), pink_noise(16000, 7)
    assert np.allclose(a, b) and abs(np.sqrt(np.mean(a ** 2)) - 1.0) < 1e-9
    assert SNR_DB == 10.0
