import glob  # パス付でファイル名を取得するために使う
import os  # ディレクトリを移動するために使う
import subprocess  # ffmpeg.exeを実行するためのモジュール

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 使用する動画があるフォルダ名
folder = ["ask", "best", "close", "day", "diary", "English", "eye", "floor",
            "glad", "guitar", "hat", "important", "just", "know", "live",
            "love", "mine", "never", "often", "pick", "plane", "right", "room",
            "see", "think", "until", "use", "will", "yet", "zoom", "テストデータ"]

# 10単語ずつ分類器を作成する。残りの20単語は不正解画像として使用する。
# 3回に分けて30単語の分類器を作成するためpositiveは3パターン用意
# 分類器を作成する単語(正解画像になる)
positive = ["ask", "best", "day", "know", "live", "love", "right", "see", "think", "use"]
# positive = ["close", "diary", "English", "eye", "floor","glad", "guitar", "hat", "important", "just"]
# positive = ["mine", "never", "often", "pick", "plane", "room", "until", "will", "yet", "zoom"]
# 入力先ディレクトリの設定
DIR_PATH = "C:\\オーラルボイス\\data\\file\\"
# 出力先ディレクトリの設定
PATH = "C:\\オーラルボイス\\classifier\\ラベル\\"


def mp4_wav(word, *file):
    "動画ファイル(mp4)を音声ファイル(wav)に変換する"
    # splitを使ってファイル名と拡張子に分けてファイル名だけをnameに入れる。
    name = [n.split(".")[0] for n in file]
    basename = os.path.splitext(os.path.basename(name[0]))[0]
    # 使用するパスを設定
    output = os.path.join(DIR_PATH, "audio", word)
    os.makedirs(output, exist_ok=True)
    input_path = os.path.join(DIR_PATH, "video", word, basename+".mp4")
    output_path = os.path.join(output, basename+".wav")

    # cmd = f'ffmpeg.exe -y -i "{input_path}" -ac 1 "{output_path}"'
    # subprocess.call(cmd)  # コマンドを実行

    # 編集(ノイズ除去)した音声ファイルのパスを指定
    wave = os.path.join(DIR_PATH, "wav", word)
    wav_path = os.path.join(wave, basename + ".wav")
    # メルスペクトログラムを作成
    create_melspectrogram(wav_path, word, basename)


def create_melspectrogram(file_path, label, fname):
    "wavファイルのデータを短時間フーリエ変換してメルスペクトログラムを作成"
    data, fs = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(2.56, 2.56)) # 画像サイズを256×256に設定
    librosa.display.specshow(log_S, sr=fs)
    plt.axis("off") # 枠と目盛りを消去

    # 作成したメルスペクトログラムを画像として保存
    save_path = os.path.join(DIR_PATH, "melspectrogram", label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))

    # 分類器で使用する画像の保存先を指定
    if label in positive:  # 正解画像の処理
        save_path = os.path.join(PATH, "positive", label)
    elif label == "テストデータ":  # テストデータの時
        save_path = os.path.join(PATH, label)
    else:  # 不正解画像の処理(テストデータではない時)
        save_path = os.path.join(PATH, "negative")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    for i in range(len(folder)):
        # 使用する動画のパス
        file_path = os.path.join(DIR_PATH, "video", folder[i], "*.mp4")

        # 正規表現でmp4ファイルだけを取得
        dir = glob.glob(file_path)

        # mp4ファイルをwavファイルにする
        for j in range(len(dir)):
            mp4_wav(folder[i], dir[j])
