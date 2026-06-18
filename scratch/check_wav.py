import wave
import numpy as np

def check(filename):
    print(f"=== Checking {filename} ===")
    try:
        with wave.open(filename, 'rb') as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
            audio = np.frombuffer(frames, dtype=np.int16)
            print("Params:", params)
            print("Min:", np.min(audio))
            print("Max:", np.max(audio))
            print("Mean:", np.mean(audio))
            print("Std:", np.std(audio))
            
            # Let's count unique values to see if it's a simple sine wave or something complex
            unique_vals = len(np.unique(audio))
            print("Unique values:", unique_vals)
    except Exception as e:
        print(f"Error checking {filename}: {e}")

check('static/audio/9a1ede43-caf2-4564-8d80-bbf44c605801.wav')
check('static/audio/44fdb975-77cd-4649-8a1e-5b557327cf29.wav')
check('static/audio/test_yo.wav')

