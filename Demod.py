import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar errores en hilos
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import time
import os

def butter_lowpass(cutoff, fs, order=6):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return b, a

def detectar_tono(bloque, tono, fs, margen=30, umbral=10):
    N = len(bloque)
    f = np.fft.rfftfreq(N, 1/fs)
    S = np.abs(np.fft.rfft(bloque * np.hanning(N)))
    idx = np.where((f >= tono - margen) & (f <= tono + margen))[0]
    if len(idx) == 0:
        return False, f, S
    energia_tono = np.max(S[idx])
    energia_promedio = np.mean(S)
    return energia_tono > energia_promedio * umbral, f, S

def siguiente_nombre():
    base = "grabacion_"
    i = 1
    while os.path.exists(f"{base}{i:03d}.wav"):
        i += 1
    return f"{base}{i:03d}.wav"

def main():
    fs = 44100
    fc = 10000
    blocksize = int(0.1 * fs)
    dur_max_mensaje = 10
    umbral_inicio = 10
    umbral_fin = 5

    sd.default.samplerate = fs
    sd.default.channels = 1

    print("🔁 Sistema activo. Esperando tono de 4000 Hz...")

    while True:
        print("🕑 Esperando 0.5 segundos antes de iniciar...")
        time.sleep(0.5)

        mensaje = []
        inicio_detectado = False
        espectro_guardado = False
        print("🎧 Escuchando en tiempo real...")

        def callback_inicial(indata, frames, time_info, status):
            nonlocal inicio_detectado, espectro_guardado, mensaje
            bloque = indata[:, 0]
            detectado, f, S = detectar_tono(bloque, 4000, fs, margen=30, umbral=umbral_inicio)
            if detectado and not inicio_detectado:
                inicio_detectado = True
                mensaje.append(bloque.copy())
                print("✅ Tono de inicio detectado.")
                if not espectro_guardado:
                    nombre_png = siguiente_nombre().replace('.wav', '_espectro.png')
                    plt.figure()
                    plt.plot(f, S)
                    plt.title("Espectro al detectar el tono de inicio (2000 Hz)")
                    plt.xlabel("Frecuencia [Hz]")
                    plt.ylabel("Magnitud")
                    plt.grid()
                    plt.savefig(nombre_png)
                    plt.close()
                    espectro_guardado = True
                    print(f"🖼️ Espectro guardado como '{nombre_png}'")
            elif inicio_detectado:
                mensaje.append(bloque.copy())

        with sd.InputStream(callback=callback_inicial, blocksize=blocksize):
            while not inicio_detectado:
                time.sleep(0.05)

            if not inicio_detectado:
                print("⌛ No se detectó tono de inicio. Reiniciando...\n")
                continue

        print("⏺️ Grabando mensaje hasta detectar tono de fin (5000 Hz)...")

        def callback_mensaje(indata, frames, time_info, status):
            nonlocal mensaje
            bloque = indata[:, 0]
            detectado_fin, _, _ = detectar_tono(bloque, 5000, fs, margen=30, umbral=umbral_fin)
            if not detectado_fin:
                mensaje.append(bloque.copy())
            else:
                print("✅ Tono de fin detectado.")
                raise sd.CallbackStop()

        with sd.InputStream(callback=callback_mensaje, blocksize=blocksize):
            try:
                sd.sleep(int(dur_max_mensaje * 1000))
            except sd.CallbackStop:
                pass

        mensaje = np.concatenate(mensaje)
        t = np.arange(len(mensaje)) / fs
        portadora = np.cos(2 * np.pi * fc * t)
        baseband = mensaje * portadora
        b, a = butter_lowpass(4000, fs)
        audio = filtfilt(b, a, baseband)
        audio /= np.max(np.abs(audio))

        plt.figure()
        plt.plot(t, audio)
        plt.title("Mensaje demodulado (dominio del tiempo)")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        plt.grid()
        nombre_senal = siguiente_nombre().replace('.wav', '_tiempo.png')
        plt.savefig(nombre_senal)
        plt.close()
        print(f"🖼️ Señal en el tiempo guardada como '{nombre_senal}'")

        output_file = nombre_senal.replace('_tiempo.png', '.wav')
        print(f"🔊 Reproduciendo mensaje demodulado...")
        sd.play(audio, fs)
        sd.wait()
        write(output_file, fs, (audio * 32767).astype(np.int16))
        print(f"💾 Audio guardado como '{output_file}'\n")

        print("🔁 Reiniciando escucha...\n")

if __name__ == '__main__':
    main()
