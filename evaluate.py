import glob
import os

import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img

# 10単語ずつ分類器を作成する。残りの20単語は不正解画像として使用する。
# 3回に分けて30単語の分類器を作成したためword_listは3パターン用意
# 分類器を作成した単語
word_list = ["ask", "best", "day", "know", "live", "love", "right", "see", "think", "use"]
# word_list =["close", "diary", "English", "eye", "floor","glad", "guitar", "hat", "important", "just"]
# word_list = ["mine", "never", "often", "pick", "plane", "room", "until", "will", "yet", "zoom"]

# 評価する単語を選択
word = word_list[7]

def main():
    # 入力画像のパラメータ
    IMG_WIGHT = 256  # 入力画像の幅
    IMG_HEIGHT = 256  # 入力画像の高さ

    # データ格納用のディレクトリパス
    PATH = "/オーラルボイス/classifier/"
    dir_path = PATH + word

    # 保存したモデル構造の読み込み
    model = model_from_json(open(dir_path + "/model.json", 'r').read())

    # 保存した学習済みの重みを読み込み
    model.load_weights(dir_path + "/weight.hdf5")

    # 画像の読み込み（正規化, 4次元配列に変換（モデルの入力が4次元なので合わせる）
    # テストする画像が入っているディレクトリ
    IMG_PATH = (PATH + 'ラベル/テストデータ')
    # 画像取得
    img_list = glob.glob(IMG_PATH + '/*.png')
    for index, i in enumerate(img_list):
        img = img_to_array(load_img(i, target_size=(IMG_WIGHT, IMG_HEIGHT)))
        img = img/255.0
        # 分類器に入力データを与えて予測（出力：各クラスの予想確率）
        y_pred = model.predict(np.array([img]))
        # 小数第4位までの表示に整形
        pred = [f'{pred:.4f}' for pred in y_pred[0]]

        # 予測結果の表示
        evaluate = os.path.basename(img_list[index])  # 評価するデータ
        print(f'{evaluate}\n{float(pred[0]) * 100:.2f}%')
        print(word)

if __name__ == '__main__':
    main()
