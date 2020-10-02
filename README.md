# 説明
発音を評価するための分類器(音声)を作成する。<br>
create_data.pyでは動画ファイル(mp4)からに音声ファイル(wav)変換して、学習させる画像(メルスペクトログラム)を作成する。<br>
augment_data.pyでは音声ファイルのデータを拡張して、正解画像と不正解画像の枚数を増やす。<br>
cnn.pyでは畳み込みニューラルネットワーク(CNN)で分類器(学習モデル)を作成する。<br>
evaluate.pyでは作成した分類器を使用して評価結果を確認する。<br>

# 実行環境
Windows10 <br>
Python 3.7.8 <br>
librosa 0.80 <br>
matplotlib 3.3.2 <br>
numpy 1.19.2 <br>
tensorflow 2.2.1 <br>
Kears 2.4.3 <br>

# インストール方法
[python3.7.8](https://www.python.org/downloads/release/python-378/)<br>
[FFmpeg](FFmpeg Builds – Builds – Zeranoe FFmpeg)

以下をコマンドプロンプトで実行する。
pip3 install librosa <br>
pip3 install matplotlib <br>
pip3 install numpy <br>
pip3 install keras <br>
pip3 install tensorflow==2.2.1 <br>
pip3 install scikit-learn <b

# 処理の流れ
Ⅰ. classifier_createフォルダを任意のフォルダにコピーする。コピーしたフォルダの中に「data」と「classifier」という名前のフォルダを作成して「data」にcreate_data.pyとaugment_data.pyを移動し、「classifier」にcnn.pyとevaluate.pyを移動する。<br>

Ⅱ. FFmpegからzip形式ファイルをダウンロードして解凍する。解凍後にbinフォルダの中からffmpeg.exeを「data」に移動する。<br>

Ⅲ. 「data」の中に「audio」,「melspectrogram」,「video」,「wav」という名前のフォルダを作成して「video」の中に動画入れる。<br>

Ⅳ. create_data.pyを実行すると「audio」に音声ファイル(wav)、「melspectrogram」に学習させる画像(メルスペクトログラム)が保存される。<br>

Ⅴ. 「classifier」の中の「ラベル」に「positive」と「negative」と「テストデータ」という名前のフォルダが作成され、「positive」には「単語名」のフォルダがある。その中に正解画像が保存される。また、「negative」には不正解画像が保存され、「テストデータ」には評価用の画像が保存される。<br>

Ⅵ. augment_data.pyを実行すると「classifier/ラベル/positive」にデータ拡張した正解画像が、「classifier/ラベルnegative」にデータ拡張した不正解画像が保存される。<br>

Ⅶ. cnn.pyを実行すると「classfier」の中に「単語名」のフォルダが作成され、その中に分類器(model.json)が保存される。<br>

Ⅷ. evaluate.pyを実行すると「classifier/ラベル/テストデータ」にある画像が評価されて点数が出力される。<br>
