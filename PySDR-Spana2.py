from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QDoubleSpinBox,
    QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QFrame
)
import pyqtgraph as pg
import numpy as np
import scipy.signal as sp_signal
import sounddevice as sd
import time
import signal
import sys
import math

# デフォルト設定
fft_size = 4096           # FFTサイズ (バッファサイズ)
num_rows = 200            # ウォーターフォールの時間軸行数
center_freq = 100e6       # 初期中心周波数 100 MHz
sample_rates = [20, 10, 5, 2, 1.92, 1, 0.5]  # MHz (USRP-RECIEVE2の1.92MHz含む)
sample_rate = 10e6
time_plot_samples = 500
gain = 50                 # 初期ゲイン 50 dB

sdr_type = "usrp"         # "usrp" または "sim"

# グローバルSDRオブジェクト
usrp = None
streamer = None
recv_buffer = None
metadata = None

def init_sdr():
    global usrp, streamer, recv_buffer, metadata, sdr_type
    if sdr_type == "usrp":
        try:
            import uhd
            # ローカル接続されたUSRP B210を初期化
            usrp = uhd.usrp.MultiUSRP()
            usrp.set_rx_rate(sample_rate, 0)
            usrp.set_rx_freq(uhd.types.TuneRequest(center_freq), 0)
            usrp.set_rx_gain(gain, 0)
            try:
                usrp.set_rx_antenna("RX2", 0)  # チャンネル0でRX2アンテナを使用
            except Exception as e:
                print(f"Antenna setup info: {e}")

            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [0]
            metadata = uhd.types.RXMetadata()
            streamer = usrp.get_rx_stream(st_args)
            recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)

            # ストリーミング開始
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            print("[PySDR-Spana2] USRP B210 Hardware successfully initialized.")
        except Exception as e:
            print(f"[PySDR-Spana2] USRP hardware connection failed ({e}). Falling back to Simulation mode.")
            sdr_type = "sim"

init_sdr()

# ==========================================
# SDR受信 & 音声復調 ワーカースレッドクラス
# ==========================================
class SDRWorker(QObject):
    def __init__(self):
        super().__init__()
        self.gain = gain
        self.sample_rate = sample_rate
        self.freq = center_freq / 1e3  # in kHz
        self.spectrogram = -50 * np.ones((fft_size, num_rows))
        self.PSD_avg = -50 * np.ones(fft_size)
        self.is_running = True
        self.frame_count = 0
        self.last_log_time = time.time()

        # 音声復調パラメータ (USRP-RECIEVE2同等)
        self.mode = "OFF"               # "OFF", "AM", "FM"
        self.volume = 0.5              # 0.0 ~ 1.0
        self.audio_rate = 48000        # 48 kHz 固定
        self.envelope_peak = 0.1       # AM AGC用ピーク初期値
        self.audio_stream = None
        self.sim_sample_idx = 0        # シミュレーション用サンプルインデックス

        # スレッドセーフなパラメータ更新用ターゲット変数
        self.target_freq = None
        self.target_gain = None
        self.target_sample_rate = None
        self.target_mode = None
        self.target_volume = None

    # PyQt シグナル
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal()

    def update_freq(self, val_khz):
        """GUIスレッドから周波数変更を指示"""
        self.target_freq = val_khz * 1e3

    def update_gain(self, val):
        """GUIスレッドからゲイン変更を指示"""
        self.target_gain = val

    def update_sample_rate(self, idx):
        """GUIスレッドからサンプリングレート変更を指示"""
        self.target_sample_rate = sample_rates[idx] * 1e6

    def update_mode(self, mode_str):
        """GUIスレッドから復調モード変更を指示 ("OFF", "AM", "FM")"""
        self.target_mode = mode_str

    def update_volume(self, val_float):
        """GUIスレッドから音量変更を指示 (0.0 ~ 1.0)"""
        self.target_volume = val_float

    def _manage_audio_stream(self):
        """音声ストリームの有効化・無効化の管理"""
        if self.mode != "OFF":
            if self.audio_stream is None:
                try:
                    self.audio_stream = sd.OutputStream(samplerate=self.audio_rate, channels=1, dtype='float32')
                    self.audio_stream.start()
                    print(f"[PySDR-Spana2] Audio OutputStream started ({self.audio_rate} Hz).")
                except Exception as e:
                    print(f"[PySDR-Spana2] Audio OutputStream error: {e}")
                    self.audio_stream = None
        else:
            if self.audio_stream is not None:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                    print("[PySDR-Spana2] Audio OutputStream stopped.")
                except Exception as e:
                    print(f"[PySDR-Spana2] Audio stop error: {e}")
                self.audio_stream = None

    def stop(self):
        self.is_running = False
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

    # メインループ
    def run(self):
        if not self.is_running:
            return

        # ワーカー内部で安全にパラメータ変更を適用
        if self.target_freq is not None:
            if sdr_type == "usrp" and usrp:
                import uhd
                usrp.set_rx_freq(uhd.types.TuneRequest(self.target_freq), 0)
            self.freq = self.target_freq / 1e3
            self.target_freq = None

        if self.target_gain is not None:
            if sdr_type == "usrp" and usrp:
                usrp.set_rx_gain(self.target_gain, 0)
            self.gain = self.target_gain
            self.target_gain = None

        if self.target_sample_rate is not None:
            if sdr_type == "usrp" and usrp:
                usrp.set_rx_rate(self.target_sample_rate, 0)
            self.sample_rate = self.target_sample_rate
            self.target_sample_rate = None

        if self.target_mode is not None:
            self.mode = self.target_mode
            self.target_mode = None
            self._manage_audio_stream()

        if self.target_volume is not None:
            self.volume = self.target_volume
            self.target_volume = None

        # USRPからサンプル受信またはシミュレーション信号作成
        if sdr_type == "usrp" and streamer:
            num_rx = streamer.recv(recv_buffer, metadata)
            if num_rx > 0:
                samples = recv_buffer[0][:num_rx]
                if len(samples) < fft_size:
                    samples = np.pad(samples, (0, fft_size - len(samples)))
            else:
                samples = np.zeros(fft_size, dtype=np.complex64)
        else:
            # シミュレーションモード: 音声復調確認用にテスト変調信号を生成
            t = (self.sim_sample_idx + np.arange(fft_size)) / self.sample_rate
            self.sim_sample_idx += fft_size
            audio_test_sig = np.sin(2 * np.pi * 1000 * t)  # 1 kHz トーン
            
            if self.mode == "AM":
                am_signal = (1.0 + 0.8 * audio_test_sig) * np.exp(2j * np.pi * 0 * t)
                noise = (np.random.randn(fft_size) + 1j * np.random.randn(fft_size)) * 0.05
                samples = am_signal + noise
            elif self.mode == "FM":
                fm_phase = 2 * np.pi * 25000 * np.cumsum(audio_test_sig) / self.sample_rate
                fm_signal = np.exp(1j * fm_phase)
                noise = (np.random.randn(fft_size) + 1j * np.random.randn(fft_size)) * 0.05
                samples = fm_signal + noise
            else:
                tone = np.exp(2j * np.pi * self.sample_rate * 0.1 * np.arange(fft_size) / self.sample_rate)
                noise = np.random.randn(fft_size) + 1j * np.random.randn(fft_size)
                samples = self.gain * tone * 0.02 + 0.1 * noise

        self.time_plot_update.emit(samples[0:time_plot_samples])

        # --- 中心周波数のAM/FM音声復調処理 (USRP-RECIEVE2同等) ---
        if self.mode != "OFF" and self.audio_stream is not None:
            if self.mode == "AM":
                # AM復調: 振幅成分の抽出とDCカット
                envelope = np.abs(samples)
                demodulated = envelope - np.mean(envelope)
                
                # AGC (ノイズフロア保障つきピーク追従)
                current_peak = np.max(np.abs(demodulated))
                self.envelope_peak = max(0.95 * self.envelope_peak + 0.05 * current_peak, 0.05)
                audio_base = demodulated / self.envelope_peak
            elif self.mode == "FM":
                # FM復調: 直交復調 (位相差 / pi で -1.0〜+1.0 に正規化)
                if len(samples) > 1:
                    demodulated = np.angle(samples[1:] * np.conj(samples[:-1])) / np.pi
                    audio_base = np.append(demodulated, 0.0)
                else:
                    audio_base = np.zeros_like(samples, dtype=np.float32)
            else:
                audio_base = None

            if audio_base is not None:
                # オーディオサンプリングレート(48kHz)へのリサンプル
                gcd = math.gcd(self.audio_rate, int(self.sample_rate))
                up = self.audio_rate // gcd
                down = int(self.sample_rate) // gcd
                audio_data = sp_signal.resample_poly(audio_base, up, down)

                # ボリューム調整とクリッピング
                audio_data = np.clip(audio_data * self.volume, -1.0, 1.0).astype(np.float32)

                # スピーカーへ出力
                try:
                    self.audio_stream.write(np.ascontiguousarray(audio_data))
                except Exception as e:
                    print(f"[PySDR-Spana2] Audio write error: {e}")

        # パワースペクトル密度 (PSD) 計算
        PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / fft_size + 1e-12)
        PSD = np.nan_to_num(PSD, nan=-50.0, posinf=-50.0, neginf=-50.0)
        self.PSD_avg = self.PSD_avg * 0.9 + PSD * 0.1
        self.freq_plot_update.emit(self.PSD_avg)

        # ウォーターフォール更新
        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1)
        self.spectrogram[:, 0] = PSD
        self.waterfall_plot_update.emit(self.spectrogram)

        # 定期的なログ出力
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 2.0:
            fps = self.frame_count / (now - self.last_log_time)
            print(f"[PySDR Spectrum Analyzer] FPS: {fps:.1f}, 周波数: {self.freq/1e3:.3f} MHz, ゲイン: {self.gain} dB, サンプルレート: {self.sample_rate/1e6:.2f} MHz, 復調: {self.mode}")
            self.frame_count = 0
            self.last_log_time = now

        if self.is_running:
            self.end_of_run.emit()


# ==========================================
# メインウィンドウ GUI クラス
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer & Demodulator (USRP B210)")
        self.resize(1280, 780)
        self.setMinimumSize(950, 550)

        self.spectrogram_min = -50
        self.spectrogram_max = 10

        layout = QGridLayout()

        # ワーカースレッドの初期化
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName('SDR_Thread')
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        # タイムドメイン表示プロット
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot.setMouseEnabled(x=False, y=True)
        time_plot.setYRange(-1.1, 1.1)
        time_plot.setMinimumHeight(140)
        time_plot_curve_i = time_plot.plot(pen=pg.mkPen('c', width=1.2))
        time_plot_curve_q = time_plot.plot(pen=pg.mkPen('m', width=1.2))
        layout.addWidget(time_plot, 1, 0)

        # タイムドメイン表示ボタン
        time_plot_auto_range_layout = QVBoxLayout()
        layout.addLayout(time_plot_auto_range_layout, 1, 1)
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda: time_plot.autoRange())
        time_plot_auto_range_layout.addWidget(auto_range_button)
        auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
        auto_range_button2.clicked.connect(lambda: time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range_button2)

        # 周波数スペクトラム表示プロット
        freq_plot = pg.PlotWidget(labels={'left': 'PSD [dB]', 'bottom': 'Frequency [MHz]'})
        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot.setMinimumHeight(150)
        freq_plot_curve = freq_plot.plot(pen=pg.mkPen('y', width=1.5))
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        freq_plot.setYRange(-80, 20)
        layout.addWidget(freq_plot, 2, 0)

        auto_range_button_freq = QPushButton('Auto Range')
        auto_range_button_freq.clicked.connect(lambda: freq_plot.autoRange())
        layout.addWidget(auto_range_button_freq, 2, 1)

        # ウォーターフォール表示レイアウト
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)

        waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
        waterfall.setMinimumHeight(160)
        imageitem = pg.ImageItem(axisOrder='col-major')
        waterfall.addItem(imageitem)
        waterfall.setMouseEnabled(x=False, y=False)
        waterfall.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        waterfall_layout.addWidget(waterfall)

        # カラーバー
        colorbar = pg.HistogramLUTWidget()
        colorbar.setImageItem(imageitem)
        colorbar.item.gradient.loadPreset('viridis')
        imageitem.setLevels((-50, 10))
        waterfall_layout.addWidget(colorbar)

        auto_range_button_wf = QPushButton('Auto Range\n(-2σ to +2σ)')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
        auto_range_button_wf.clicked.connect(update_colormap)
        layout.addWidget(auto_range_button_wf, 3, 1)

        # --- コントロールパネル領域 (SDR設定 & 音声復調設定) ---
        controls_layout = QHBoxLayout()

        # 1. SDR ハードウェア設定グループボックス
        sdr_group = QGroupBox("SDR ハードウェア設定 (Hardware Controls)")
        sdr_grid = QGridLayout()

        # 周波数スライダー & SpinBox
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(70000, int(6e6))  # 70 MHz 〜 6000 MHz (kHz単位)
        freq_slider.setValue(int(center_freq / 1e3))
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        freq_slider.setTickInterval(int(500e3))

        freq_spinbox = QDoubleSpinBox()
        freq_spinbox.setRange(70.0, 6000.0)
        freq_spinbox.setDecimals(3)
        freq_spinbox.setSingleStep(1.0)
        freq_spinbox.setSuffix(" MHz")
        freq_spinbox.setValue(center_freq / 1e6)

        def update_freq_display(val_khz):
            f_center = val_khz * 1e3 / 1e6
            half_bw = self.worker.sample_rate / 2e6
            freq_plot.setXRange(f_center - half_bw, f_center + half_bw)
            waterfall.setXRange(f_center - half_bw, f_center + half_bw)

        def on_freq_slider_changed(val_khz):
            val_mhz = val_khz / 1e3
            freq_spinbox.blockSignals(True)
            freq_spinbox.setValue(val_mhz)
            freq_spinbox.blockSignals(False)
            self.worker.update_freq(val_khz)
            update_freq_display(val_khz)

        def on_freq_spinbox_changed(val_mhz):
            val_khz = int(val_mhz * 1e3)
            freq_slider.blockSignals(True)
            freq_slider.setValue(val_khz)
            freq_slider.blockSignals(False)
            self.worker.update_freq(val_khz)
            update_freq_display(val_khz)

        freq_slider.valueChanged.connect(on_freq_slider_changed)
        freq_spinbox.valueChanged.connect(on_freq_spinbox_changed)

        sdr_grid.addWidget(QLabel("周波数:"), 0, 0)
        sdr_grid.addWidget(freq_slider, 0, 1)
        sdr_grid.addWidget(freq_spinbox, 0, 2)

        # ゲインスライダー
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 76)
        gain_slider.setValue(gain)
        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText(f"Gain: {val} dB")
            self.worker.update_gain(val)
        gain_slider.valueChanged.connect(update_gain_label)
        update_gain_label(gain_slider.value())

        sdr_grid.addWidget(QLabel("ゲイン:"), 1, 0)
        sdr_grid.addWidget(gain_slider, 1, 1)
        sdr_grid.addWidget(gain_label, 1, 2)

        # サンプリングレート ComboBox
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
        default_sr_idx = sample_rates.index(10) if 10 in sample_rates else 0
        sample_rate_combobox.setCurrentIndex(default_sr_idx)
        sample_rate_label = QLabel()
        def update_sample_rate_label(idx):
            sample_rate_label.setText(f"Sample Rate: {sample_rates[idx]} MHz")
            self.worker.update_sample_rate(idx)
            update_freq_display(freq_slider.value())
        sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_combobox.currentIndex())

        sdr_grid.addWidget(QLabel("サンプリングレート:"), 2, 0)
        sdr_grid.addWidget(sample_rate_combobox, 2, 1)
        sdr_grid.addWidget(sample_rate_label, 2, 2)

        sdr_group.setLayout(sdr_grid)
        controls_layout.addWidget(sdr_group, stretch=3)

        # 2. 中心周波数 音声復調設定グループボックス
        audio_group = QGroupBox("中心周波数 音声復調 (Audio Demodulation)")
        audio_grid = QGridLayout()

        mode_combobox = QComboBox()
        mode_combobox.addItems(["OFF (Mute)", "AM (振幅変調)", "FM (周波数変調)"])
        mode_combobox.setCurrentIndex(0)

        def on_mode_changed(idx):
            modes = ["OFF", "AM", "FM"]
            selected_mode = modes[idx]
            self.worker.update_mode(selected_mode)

        mode_combobox.currentIndexChanged.connect(on_mode_changed)

        audio_grid.addWidget(QLabel("復調方式:"), 0, 0)
        audio_grid.addWidget(mode_combobox, 0, 1)

        vol_label = QLabel("音量 (Volume): 50%")
        vol_slider = QSlider(Qt.Orientation.Horizontal)
        vol_slider.setRange(0, 100)
        vol_slider.setValue(50)

        def on_vol_changed(val):
            vol_label.setText(f"音量 (Volume): {val}%")
            self.worker.update_volume(val / 100.0)

        vol_slider.valueChanged.connect(on_vol_changed)

        audio_grid.addWidget(vol_label, 1, 0, 1, 2)
        audio_grid.addWidget(vol_slider, 2, 0, 1, 2)

        audio_group.setLayout(audio_grid)
        controls_layout.addWidget(audio_group, stretch=2)

        layout.addLayout(controls_layout, 4, 0, 1, 2)

        # 全体をスクロール可能にする (画面サイズオーバー防止)
        scroll_content = QWidget()
        scroll_content.setLayout(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)

        self.setCentralWidget(scroll_area)

        # シグナルとスロットのコールバック
        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)

        def freq_plot_callback(PSD_avg):
            f_center = freq_slider.value() * 1e3
            f = np.linspace(f_center - self.worker.sample_rate / 2.0, f_center + self.worker.sample_rate / 2.0, fft_size) / 1e6
            freq_plot_curve.setData(f, PSD_avg)

        def waterfall_plot_callback(spectrogram):
            imageitem.setImage(spectrogram, autoLevels=False)
            f_center = freq_slider.value() * 1e3 / 1e6
            half_bw = self.worker.sample_rate / 2e6
            freq_min = f_center - half_bw
            freq_max = f_center + half_bw
            imageitem.setRect(freq_min, 0, freq_max - freq_min, num_rows)

            sigma = np.std(spectrogram)
            mean = np.mean(spectrogram)
            self.spectrogram_min = mean - 2 * sigma
            self.spectrogram_max = mean + 2 * sigma

        def end_of_run_callback():
            if self.worker.is_running:
                QTimer.singleShot(0, self.worker.run)

        self.worker.time_plot_update.connect(time_plot_callback)
        self.worker.freq_plot_update.connect(freq_plot_callback)
        self.worker.waterfall_plot_update.connect(waterfall_plot_callback)
        self.worker.end_of_run.connect(end_of_run_callback)

        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()

    def closeEvent(self, event):
        """ウィンドウ終了時のクリーンアップ"""
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait(1000)
        if streamer:
            try:
                import uhd
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                streamer.issue_stream_cmd(stop_cmd)
            except Exception:
                pass
        event.accept()

# ==========================================
# メインエントリーポイント
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec())
