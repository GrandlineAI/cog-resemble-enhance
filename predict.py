# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import torchaudio
from typing import List
from resemble.resemble_enhance.enhancer.inference import denoise, enhance
import subprocess
ffmpeg = "/usr/local/bin/ffmpeg"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        

    def predict(
        self,
        input_file: Path = Input(description="Input video/audio file"),
        solver: str = Input(
            description="Solver to use",
            default="Midpoint",
            choices=["Midpoint", "RK4", "Euler"]
        ),
        number_function_evaluations: int = Input(
            description="CFM Number of function evaluations to use",
            default=64, ge=1, le=128
        ),
        prior_temperature: float = Input(
            description="CFM Prior temperature to use",
            default=0.5, ge=0, le=1.0
        ),
        denoise_flag: bool = Input(
            description="Denoise the audio",
            default=False
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        solver = solver.lower()
        nfe = int(number_function_evaluations)
        lambd = 0.9 if denoise_flag else 0.1

        # Run ffprobe command to get the file information
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(input_file)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Check if the file is a video file
        if result.stdout:
            print("This is a video file.")
            # Separate the video to audio and video
            subprocess.run(["ffmpeg", "-y", "-i", str(input_file), "-vn", "-acodec", "copy", "output.m4a"])
            subprocess.run(["ffmpeg", "-y", "-i", str(input_file), "-an", "-vcodec", "copy", "output.mp4"])
            dwav, sr = torchaudio.load(str("output.m4a"))
        else:
            print("This is an audio file.")
            dwav, sr = torchaudio.load(str(input_file))

        dwav = dwav.mean(dim=0)

        wav1, new_sr1 = denoise(dwav, sr, device)
        wav2, new_sr2 = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=prior_temperature)

        wav1 = wav1.unsqueeze(0)
        wav2 = wav2.unsqueeze(0)

        outputs = []
        output_path1 = "/tmp/output-denoised.wav"
        output_path2 = "/tmp/output-enhanced.wav"
        torchaudio.save(output_path1, wav1, new_sr1)
        torchaudio.save(output_path2, wav2, new_sr2)
        outputs.append(Path(output_path1))
        outputs.append(Path(output_path2))

        # If the file is a video file, merge the video and the audio
        if result.stdout:
            # Merge the video and the denoised audio
            subprocess.run(["ffmpeg", "-y", "-i", "output.mp4", "-i", output_path1, "-c:v", "copy", "-c:a", "aac", "output-denoised.mp4"])

            # Merge the video and the enhanced audio
            subprocess.run(["ffmpeg", "-y", "-i", "output.mp4", "-i", output_path2, "-c:v", "copy", "-c:a", "aac", "output-enhanced.mp4"])

            # Append the paths to the output files
            outputs.append(Path("output-denoised.mp4"))
            outputs.append(Path("output-enhanced.mp4"))

        return outputs
