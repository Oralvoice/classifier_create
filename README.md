# 説明
発音を評価するための分類器(音声)を作成する。<br>
create_data.pyでは動画ファイル(mp4)からに音声ファイル(wav)変換して、学習させる画像(メルスペクトログラム)を作成する。<br>
augment_data.pyでは音声ファイルのデータを拡張して、正解画像と不正解画像の枚数を増やす。<br>
cnn.pyでは畳み込みニューラルネットワーク(CNN)で分類器(学習モデル)を作成する。<br>
evaluate.pyでは作成した分類器を使用して評価結果を確認する。<br>

