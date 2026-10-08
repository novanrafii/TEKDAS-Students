"""Grounded LLM prompts for student-retention analysis.

Prompt-only check: python llm.py --prompt-only
Requires Ollama for generation: ollama serve; ollama pull llama3.2
"""
import argparse
import os
from typing import Any

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(
    student_profile: dict[str, Any],
    predicted_target: str,
    probabilities: dict[str, float],
    model_factors: list[dict[str, Any]] | None = None,
) -> str:
    """Create a prompt that separates observed facts from model prediction."""
    profile = {
        key: student_profile[key]
        for key in [
            "Age at enrollment",
            "Course",
            "Daytime/evening attendance",
            "Scholarship holder",
            "Tuition fees up to date",
            "Debtor",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
        ]
        if key in student_profile
    }
    return f"""Anda adalah asisten analisis data pendidikan.
Gunakan hanya fakta yang tersedia di bawah. Jangan mengarang kondisi, alasan, atau latar belakang mahasiswa.
Prediksi model adalah estimasi, bukan fakta dan bukan bukti sebab-akibat. Hindari menyimpulkan sebab dropout dari korelasi.

PROFIL TERPILIH (kode kategori tetap berupa kode):
{profile}

OUTPUT MODEL:
Prediksi kelas: {predicted_target}
Probabilitas tiap kelas: {probabilities}

FAKTOR GLOBAL MODEL (bukan penjelasan kausal individual):
{model_factors or 'Tidak tersedia.'}

TUGAS:
Tulis dalam bahasa Indonesia secara ringkas:
1. Interpretasi hasil prediksi dan tingkat ketidakpastiannya.
2. Maksimal dua pengamatan yang langsung didukung profil atau probabilitas.
3. Dua langkah tindak lanjut yang etis dan dapat diverifikasi oleh pihak kampus.
4. Batasan analisis, termasuk bahwa kode kategori perlu kamus data dan prediksi tidak menetapkan penyebab.
Gunakan poin-poin dan jangan menyatakan mahasiswa pasti akan dropout/lulus.""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to a locally running Ollama server."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Paket Ollama belum terpasang. Jalankan pip install -r requirements.txt.") from exc
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Uji prompt analisis retensi mahasiswa.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    sample = {
        "Age at enrollment": 20,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 3,
        "Curricular units 1st sem (grade)": 11.5,
    }
    prompt = build_prompt(sample, "Dropout", {"Dropout": 0.62, "Enrolled": 0.21, "Graduate": 0.17})
    if args.prompt_only:
        print(prompt)
        return
    print(call_ollama(prompt, args.model))


if __name__ == "__main__":
    main()
