import numpy as np
import librosa
import random
import glob
import os
import matplotlib.pyplot as plt
import librosa.display

# 入力先ディレクトリの設定
INPUT = "C:\\オーラルボイス\\data\\file\\wav\\"
# 出力先ディレクトリの設定
OUTPUT = ("C:\\オーラルボイス\\classifier\\ラベル\\positive\\")

# 10単語ずつ分類器を作成する。残りの20単語は不正解画像として使用する。
# 3回に分けて30単語の分類器を作成するためlabelsは3パターン用意
# 出力するファイル名
labels = ["ask", "best", "day", "know", "live", "love", "right", "see", "think", "use"]
# labels= ["close", "diary", "English", "eye", "floor","glad", "guitar", "hat", "important", "just"]
# labels = ["mine", "never", "often", "pick", "plane", "room", "until", "will", "yet", "zoom"]
def rand_nodup(a, b, num):
    """重複しない乱数を生成"""
    ns = []
    while len(ns) < num:
        random.seed()
        n = round(random.uniform(a, b), 3)
        if n not in ns:
            ns.append(n)
    return ns


def load_audio_file(file_path):
    """音声ファイルを読み込む"""
    input_length = 40000
    data = librosa.core.load(file_path)[0]
    if len(data) > input_length:
        data = data[: input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def add_white_noise(data, rate):
    """データ拡張：ノイズを乗せる"""
    rate = np.random.choice(rate)/400
    return data + rate*np.random.randn(len(data))


def shift_sound(data, rate):
    """データ拡張：時間をずらす"""
    rate = np.random.choice(rate)
    return np.roll(data, int(len(data)//rate))


def stretch_sound(data, rate):
    """データ拡張：音を伸ばす"""
    input_length = len(data)
    rate = np.random.choice(rate)
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        return data[: input_length]
    else:
        return np.pad(data, (0, max(0, input_length - len(data))), "constant")


def combination(data):
    """データ拡張：ノイズ、シフト、ストレッチをランダムに組み合わせる"""
    rate = rand_nodup(0.8, 1.0, 1)
    data = add_white_noise(data, rate)
    if np.random.choice((True, False)):
        data = shift_sound(data, rate)
    else:
        data = stretch_sound(data, rate)
    return data


def augment_data(filename, aug=None):
    """データを拡張する"""
    for i in range(len(labels)):
        # 拡張する音声ファイルの読み込み
        file = glob.glob(os.path.join(INPUT, labels[i], "*.wav"))
        # 読み込んだ音声ファイルのデータを拡張
        for j in range(len(file)):
            wava_data = file[j]
            basename = os.path.splitext(os.path.basename(wava_data))[0]
            data, fs = librosa.load(wava_data, sr=None)

            if aug is combination:
                wave = aug(data=data)
                # ファイル名を作成して保存
                fname = f"{basename}_{filename}"
                save_data(wave, labels[i], fname, fs)
            else:
                for k in range(6):
                    rates = rand_nodup(0.8, 1.0, 6)
                    wave = aug(data=data, rate=rates)
                    # ファイル名を作成して保存
                    fname = f"{basename}_{filename}_{k}"
                    save_data(wave, labels[i], fname, fs)


def save_data(data, label, name, fs):
    "拡張した音声ファイルのデータを使用してメルスペクトログラムを作成"
    # メルスペクトログラムを算出
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(2.56, 2.56)) # 画像サイズを256×256に設定
    librosa.display.specshow(log_S, sr=fs)
    # 枠と目盛りを消去
    plt.axis("off")

    # 作成したメルスペクトログラムを画像として保存
    save_path = os.path.join(OUTPUT, label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{name}.png"))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # データ拡張(ランダム)
    augment_data("random", aug=combination)
    # データ拡張(ノイズをのせる)
    augment_data("white_noise", aug=add_white_noise)
    # データ拡張(時間をずらす)
    augment_data("sound_shift", aug=shift_sound)
    # データ拡張(音を伸ばす)
    augment_data("sound_stretch", aug=stretch_sound)
