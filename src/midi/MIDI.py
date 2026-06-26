import subprocess
import os

class MIDIHandler():

    def __init__():
        pass

    def synthesize():
        pass

    def tokenize():
        # just call MIDITokenizer.tokenize()
    
    def detokenize():
        # just call MIDITokenizer.detokenize()

    # any other methods 


# Some sort of it
def synthesize(data_dir: str) -> None:
    """
    Synthesize all .mid files in data_dir using the nesmdb-env Docker container.

    Args:
        data_dir: Path to the directory containing .mid files to convert.
    """
    data_dir = os.path.abspath(data_dir)

    bash_script = """
    find /data -type f -name "*.mid" | while read -r file; do
        echo "Converting: $file"
        nesmdb_convert midi_to_wav "$file" --out_dir /data/wavs_from_midi/
    done
    echo "Midi Batch conversion complete!"
    """

    command = [
        "docker", "run", "--rm",
        "--entrypoint", "bash",
        "-v", f"{data_dir}:/data",
        "nesmdb-env",
        "-c", bash_script,
    ]

    print(f"Starting synthesis for directory: {data_dir}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Docker command failed with exit code {result.returncode}.\n"
            f"Output:\n{result.stdout}"
        )

    print("Synthesis complete. WAV files saved to:", os.path.join(data_dir, "wavs_from_midi"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python synthesize.py <data_dir>")
        sys.exit(1)
    synthesize(sys.argv[1])