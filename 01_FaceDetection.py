# -*- coding: utf-8 -*-

import cv2  #オープンCVのモジュールをインポートする

#メイン関数
if __name__ == '__main__':
    try:
        #カスケード識別器のパス
#        cascade_path = "haarcascade_frontalface_alt.xml"
        cascade_path = "haarcascade_frontalface_default.xml"

        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        #ビデオキャプチャーを取得する:0はデバイス番号
        capture = cv2.VideoCapture(0)

        #ビデオキャプチャーを開けていない場合は例外を
        if capture.isOpened() is False:
            raise IOError("VideoCapture could not open.")

        #メインループ
        while True:
            # ビデオキャプチャからフレームを取得
            ret, image = capture.read()

            if ret == False:
                continue

            #カスケード識別器にて物体検出（顔を含む矩形を取得する）
            facerect = cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

            #検出した矩形を描写する
#            color = (255, 255, 255)     #白色
            color = (0, 0, 255)     #赤色
            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), color, thickness=2)

            #ウィンドウに表示する
            cv2.imshow("FaceDetection", image)

            #wait[ms]
            cv2.waitKey(10)
    except KeyboardInterrupt  :         #Ctl+Cが押されたらループを終了
        print("\nCtl+C")
    except Exception as e:              #その他の例外が発生した場合は、
        print(str(e))                   #例外処理の内容をコンソールに表示
    finally:
        capture.release()               #Videoキャプチャをリリースする
        cv2.destroyAllWindows()         #すべてのOpenCVウィンドウをクローズする
        for i in range (1,5):
            cv2.waitKey(10)             #少し待たないとウィンドウがクローズされない
        print("\nexit program")         #プログラムが終了することを表示する
