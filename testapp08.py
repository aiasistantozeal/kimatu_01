### 事前準備 ###
#
# AiPro2022フォルダに移動して…
#
# python -m venv .tfflow    ← 新しい仮想環境「.tfflow」を生成する
#
# .tfflow\Scripts\activate.bat    ← 新しい仮想環境「.tfflow」を起動する
#
# (.tfflow) > python -m pip install -r requirements.txt   ← ライブラリの一括インストール
# ※一括インストールは時間がかかりますので(10分～？）休み時間の前などに実行するとよいでしょう
#
# requirements.txt の内容
#   streamlit             ← 毎度、おなじみ
#   typing_extensions     ← これがないとエラーが出ることがあるので一応
#   numpy                 ← 毎度、おなじみ
#   pandas                ← 毎度、おなじみ
#   tensorflow-cpu        ← 深層学習用ライブラリ（容量の関係で今回はCPU版を採用）
#
# (.tfflow) > streamlit run testapp04.py    ← Webアプリ（testapp08.py）を起動

### Teachable Machineでモデルを作成する方法（中級以上） ###
#
#（Teachable Machineの詳細は、「AI理論」の授業で配布された「認識編Ⅰ」を参照）
# 
# ※推奨ブラウザ：Google Chrome（他のブラウザだと、ファイルのD&Dができないかも）
# 
# 1. Teachable Machineにアクセスする → 使ってみる → 画像プロジェクト → 標準の～
#   https://teachablemachine.withgoogle.com/
#   
# 2. 各Classの「アップロード」ボタンからデータを追加する
# 
# 3. トレーニングを実行する
#
# 4. モデルをエクスポートする　→　Tensorflowタブ → Keras → モデルをダウンロード
#
# 5. モデルのファイル（converted_keras.zip）がダウンロードされる
# 
# 6. converted_keras.zip に含まれる「keras_model.h5」を、
#    この testapp08.py ファイルと「同じフォルダ」に格納する
#
# 7. 準備OK！
# 
# （以後、モデルを作り直すたびに、エクスポート → .h5ファイルの上書きをしてください）
#リンク(本物の車、フェイクの車)
#https://teachablemachine.withgoogle.com/models/C22uLcNUd/


# ライブラリのインポート
from ssl import SSLSession
from sys import _xoptions
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# 画像(img)が属するクラスを推論する関数（'weights_file'は、モデルのファイル名）
# 以下、Teachable Machineのエクスポート時に自動生成されるコードをコピペする(済)
def teachable_machine_classification(img, weights_file):

    # モデルの読み込み
    model = load_model(weights_file)

    # kerasモデルに投入するのに適した形状の配列を作成する。
    # 配列に入れることができる画像の「長さ」または枚数は
    # shapeタプルの最初の位置（この場合は1）で決まる。
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # これを画像へのパスに置き換える
    # image = Image.open(img)
    image = img

    # Teachable Machineと同じ方法で、224x224にリサイズする。
    # 少なくとも224x224になるように画像をリサイズし、中心から切り取る。
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # 画像をnumpyの配列に変換する
    image_array = np.asarray(image)

    # 画像の正規化
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 画像を配列に読み込む
    data[0] = normalized_image_array

    # 推論を実行する
    prediction = model.predict(data)

    # 推論結果をメインモジュールに戻す
    return prediction.tolist()[0]

#サンプル画像の答え合わせ
def quiz(Ai_answer: str, You_answer: str):
    st.write("----------------------------")
    st.write("★サンプル画像の答え合わせ★")
    quiz_answer = ["本物","フェイク","フェイク","本物","フェイク","本物","本物","フェイク"]
    max_value= 8
    min_value= 0
    slider=st.slider("サンプル画像の番号を選択してください", min_value, max_value, min_value, 1)
    st.write("例：2.jpg →　2  ")
    if slider == 0:
        st.write("[※1~8を選択]")
    else:
        st.subheader(f'答え：[この画像は{quiz_answer[slider -1]}でした！]')

        Ai_answer_point = 0
        You_answer_point = 0

        if Ai_answer == quiz_answer[slider -1]:
            Ai_answer_point = 1
            st.session_state['Total_ai_point'] += 1
        else:
            pass
        if You_answer == quiz_answer[slider -1]:
            You_answer_point = 1
            st.session_state['Total_you_point'] += 1
        else:
            pass

        if Ai_answer_point > You_answer_point:
            st.subheader("不正解！AIの勝利！")
        elif Ai_answer_point < You_answer_point:
            st.subheader("正解！あなたの勝利！")
        elif ((Ai_answer_point == 1) and (You_answer_point == 1)):
            st.subheader("両者正解！流石です！")
        elif ((Ai_answer_point == 0) and (You_answer_point == 0)):
            st.subheader("両者不正解！あらら...")
        else:
            st.subheader("エラー")

     # 現在のトータル戦績の状況を表示
    st.write("■□■□■ 戦績（トータル） ■□■□■")
    win_you = st.session_state['Total_you_point']
    win_ai = st.session_state['Total_ai_point']

    # トータルポイントの差を表示
    if win_you > win_ai :
        st.write("★あなたの優勢です!★")
        win_you_dif = f'+{win_you - win_ai}'
        win_ai_dif = f'-{win_you - win_ai}'
    elif win_ai > win_you:
        st.write("◆Aiの優勢です◆")
        win_you_dif = f'-{win_ai - win_you}'
        win_ai_dif = f'+{win_ai - win_you}'
    else:
        if ((win_you == 0) and (win_ai == 0)):
            st.write("ー－－－ー")
        else:
            st.write("～両者互角！～")
        win_you_dif = "±0"
        win_ai_dif = "±0"
        
    
    # 勝敗履歴の表示
    st.write(f'あなたの正解数：{win_you}回   {win_you_dif}')
    st.write(f'AIの正解数：{win_ai}回   {win_ai_dif}')
    



# メインモジュール
def main():

    #セッションの初期化
    if 'Total_ai_point' not in st.session_state:
        st.session_state['Total_ai_point'] = 0
    if 'Total_you_point' not in st.session_state:
        st.session_state['Total_you_point'] = 0

    # タイトルの表示
    st.title("車のフェイク画像を見分けられるかAIと勝負をします。")

    # アップローダの作成
    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")
  
    # 自分の解答
    You_answer = st.radio(label='あなたは「本物の車画像」と「フェイクの車画像」のどちらだと思いますか？',
                 options=('本物', 'フェイク'),
                 index=0,
                 horizontal=True,)
    
    # 画像がアップロードされた場合...
    if uploaded_file is not None:
            # 画像を画面に表示
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            #st.write("")
            #st.write("Classifying...")

            # teachable_machine_classification関数に画像を引き渡してクラスを推論する
            prediction = teachable_machine_classification(image, 'keras_model.h5')        
            #st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

            classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
            #st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

            # 推論の確率を小数点以下3桁で丸め×100(%に変換)
            pred0 = round(prediction[0],3) * 100  # 本物の確率(%)
            pred1 = round(prediction[1],3) * 100  # フェイクの確率(%)

            # ユーザの選択の表示
            st.write("----------------------------")
            st.write(f'あなた：この車の画像は{You_answer}です！')

            # 推論で得られたクラス番号(初期値は0)によって出力結果を分岐
            if classNo == 0:
                st.subheader(f"AI :これは{pred0}％の確率で「本物の(実在する)車の画像」です！")
                Ai_answer = "本物"
            else:
                st.subheader(f"AI :これは{pred1}％の確率で「フェイク画像の車」です！")
                Ai_answer = "フェイク"

            # 正解を表示（サンプル画像の答え合わせ）
            quiz(Ai_answer,You_answer)
    else:
        st.write("※画像をアップロードしてください!")




# mainの起動
if __name__ == "__main__":
    main()


#=============================================================================#
# 氏名：佐藤光
# 学科：AIシステム科
# 学年：２年
# コメント：現在の画像生成技術で作成されたフェイク画像を見分けるのは人間、AI共に難しいことがわかりました。
#
#=============================================================================#
