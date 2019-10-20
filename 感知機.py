# -*- coding: utf-8 -*-
import numpy as np;
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from PyQt5 import QtWidgets

#Canvas
class Figure_Canvas(FigureCanvas):

    def __init__(self, parent=None, width=3, height=1.6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=100)  

        FigureCanvas.__init__(self, fig) 
        self.setParent(parent)

        self.axes = fig.add_subplot(111) 

    def test(self, data, w, n):
        #Plot point 
        color = {
                    1:'red',
                    2:'blue',
                    3:'yellow',
                    4:'green'
                }
        for i in range(n):
            self.axes.scatter(data[i, 1], data[i, 2], color=color[data[i, 3]], s = 15, alpha=0.8)
        #Plot line
        a = np.arange(-10, 10, 0.1)
        b =( w[0] + -1*w[1]*a ) / w[2] 
        self.axes.plot(a, b)
        self.axes.set_xlim(max(data[:, 1]) + 1, min(data[:, 1]) - 1)
        self.axes.set_ylim(max(data[:, 2]) + 1, min(data[:, 2]) - 1)
    
    #2類以上用
    def test2(self, data, w, w2, n):
        #Plot point 
        color = {
                    0:'purple',
                    1:'red',
                    2:'blue',
                    3:'yellow',
                    4:'green'
                }
        for i in range(n):
            self.axes.scatter(data[i, 1], data[i, 2], color=color[data[i, 3]], s = 15, alpha=0.8)
        #Plot line
        a = np.arange(-2, 2, 0.1)
        b =( w[0] + -1*w[1]*a ) / w[2]
        b2 =( w2[0] + -1*w2[1]*a ) / w2[2]
        self.axes.plot(a, b)
        self.axes.plot(a, b2)
        self.axes.set_xlim(max(data[:, 1]) + 1, min(data[:, 1]) - 1)
        self.axes.set_ylim(max(data[:, 2]) + 1, min(data[:, 2]) - 1)

#Training
class Train():

    def __init__(self):
        self.fileName = ""
        self.rate = 0.0
        self.count = 0
        self.count = 0
        self.progress_bar = QtWidgets.QProgressBar()
        self.train_acc_text = ""
        self.test_acc_text = ""
        self.weight = ""

    def set(self, fileName, rate, rnd, count):
        self.fileName = fileName
        self.rate = rate
        self.round = rnd
        self.count = count

    def run(self):
        self.progress_bar.setValue(0)
        
        #Read data
        fi=open(self.fileName, 'r')
        #Set iteration
        it = self.round
        #Set learning rate
        lr = self.rate
        lr_o = self.rate
        #計算維度
        line=fi.readline()
        line = line.split()
        dim = len(line)        #資料維度
        #讀取數據用
        data = np.empty(shape=[0, dim + 1])
        #把第一筆加上去
        line = np.hstack((['-1'], line))
        data = np.row_stack((data, line))
        add = 0
        #資料筆數
        n = 1
        while 1:
            line=fi.readline()
            if line=="":
                break
            line = line.split()
            line = np.hstack((['-1'], line))
            data = np.row_stack((data, line))
            if int(data[-1, -1]) == 0:
                add = 1
            n = n + 1
        data = data.astype(np.float)
        #計算類數
        myset = set(data[:, dim])
        cls = len(myset)
        #處理label
        if add == 1 and cls ==2:
            for i in range(n):
                data[i, -1] = data[i, -1] + 1 
        #切割資料
        train_data = np.empty(shape=[0, dim + 1])
        test_data = np.empty(shape=[0, dim + 1])
        test_num = random.sample(range(len(data)), k = int(len(data) / 3) )
        for i in range(len(data)):
            if i in test_num:
                test_data = np.row_stack((test_data, data[i]))
            else:
                train_data = np.row_stack((train_data, data[i]))
        #Set Weight 
        w = [random.uniform(-1, 1)] * (dim)
        w2 = [random.uniform(-1, 1)] * (dim)    #第二科神經元用
        #best accuracy
        b_acc = 0
        test_acc = 0
        
        if cls == 2 :
            #Training
            for i in range(it):
                for k in range(len(train_data)):
                    y = np.dot(train_data[k, 0:3], w)
                    if sgn(y) == 2 and train_data[k, 3] == 1 :
                        w = w - lr*train_data[k, 0:3]
                    elif sgn(y) == 1 and train_data[k, 3] == 2 :
                        w = w + lr*train_data[k, 0:3]
                #計算最佳訓練精準度
                cnt = 0
                for k in range(len(train_data)):
                    y = np.dot(train_data[k, 0:3], w)
                    if sgn(y) == train_data[k, 3]:
                        cnt = cnt + 1
                acc = cnt/len(train_data)
                if acc > b_acc :
                    b_acc = acc
                    b_w = w
                ###################
                #調整訓練速率
                lr = lr_o * ( ( it - i ) / it)
                self.progress_bar.setValue((i/it)*100) 
            #計算測試精準度
            cnt = 0
            for k in range(len(test_data)):
                y = np.dot(test_data[k, 0:3], w)
                if sgn(y) == test_data[k, 3]:
                    cnt = cnt + 1
            test_acc = cnt/len(test_data)
            #顯示權重
            self.weight = str(b_w)
            self.progress_bar.setValue(100)
            
        elif cls > 2 :
            #Training
            for i in range(it):
                for k in range(len(train_data)):
                    y = np.dot(train_data[k, 0:dim], w)
                    y2 = np.dot(train_data[k, 0:dim], w2)
                    
                    if sgn_bin(y) == 1 and int((train_data[k, dim]-1)/2) == 0 :
                        w = w - lr*train_data[k, 0:dim]
                    elif sgn_bin(y) == 0 and int((train_data[k, dim]-1)/2) == 1 :
                        w = w + lr*train_data[k, 0:dim]
                        
                    if sgn_bin(y2) == 1 and int((train_data[k, dim]-1)%2) == 0 :
                        w2 = w2 - lr*train_data[k, 0:dim]
                    elif sgn_bin(y2) == 0 and int((train_data[k, dim]-1)%2) == 1 :
                        w2 = w2 + lr*train_data[k, 0:dim]
                #計算最佳訓練精準度
                cnt = 0
                b_acc = 0
                for k in range(len(train_data)):
                    y = np.dot(train_data[k, 0:dim], w)
                    y2 = np.dot(train_data[k, 0:dim], w2)
                    if sgn_bin(y) == int((train_data[k, dim]-1)/2) and sgn_bin(y2) == int((train_data[k, dim]-1)%2):
                        cnt = cnt + 1
                acc = cnt/len(train_data)
                if acc > b_acc :
                    b_acc = acc
                    b_w = w
                    b_w2 = w2
                #################
                #調整訓練速率
                lr = lr_o * ( ( it - i ) / it)
                self.progress_bar.setValue((i/it)*100)
            #計算測試訓練精準度
            cnt = 0
            for k in range(len(test_data)):
                y = np.dot(test_data[k, 0:dim], w)
                y2 = np.dot(test_data[k, 0:dim], w2)
                if sgn_bin(y) == int((test_data[k, dim]-1)/2) and sgn_bin(y2) == int((test_data[k, dim]-1)%2):
                    cnt = cnt + 1
            test_acc = cnt/len(test_data)
            #顯示權重
            self.weight = str(b_w) + str(b_w2)
            self.progress_bar.setValue(100)
        
        #顯示精準度
        self.train_acc_text = str(b_acc * 100)
        self.test_acc_text = str(test_acc * 100)
        '''
        #Plot point 
        color = {
                    1:'red',
                    2:'blue',
                    3:'yellow',
                    4:'green'
                }
        for i in range(len(test_data)):
            plt.scatter(test_data[i, 1], test_data[i, 2], color=color[test_data[i, dim]], s = 15, alpha=0.8)
        #Plot line
        a = np.arange(-2, 2, 0.1)
        b =( b_w[0] + -1*b_w[1]*a ) / b_w[2]
        plt.plot(a, b)
        if cls > 2 :
            b2 = ( b_w2[0] + -1*b_w2[1]*a ) / b_w2[2]
            plt.plot(a, b2)
        plt.ylim((-2.5, 2.5))
        plt.show()
        '''
        
        self.train_dr = Figure_Canvas()
        self.test_dr = Figure_Canvas()
        if cls == 2 :
            self.train_dr.test(train_data, b_w, len(train_data))
            self.test_dr.test(test_data, b_w, len(test_data))
        elif cls > 2 :
            self.train_dr.test2(train_data, b_w, b_w2, len(train_data))
            self.test_dr.test2(test_data, b_w, b_w2, len(test_data))
          
        fi.close()
        
    def get_test_pic(self):
        return self.test_dr
    
    def get_train_pic(self):
        return self.train_dr

#GUI
class Input(QtWidgets.QWidget):

    def __init__(self, parent = None):

        super().__init__(parent)
        self.progress_bar = []
        self.fileName = ""
        self.rate = 0.0
        self.round = 0
        self.count = 0

        self.layout = QtWidgets.QFormLayout()
        self.Label1 = QtWidgets.QLabel("File name")
        self.tmp1 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label1, self.tmp1)

        self.Label2 = QtWidgets.QLabel("Learning rate")
        self.tmp2 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label2, self.tmp2)
        
        self.Label3 = QtWidgets.QLabel("Round")
        self.tmp3 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label3, self.tmp3)

        self.btn = QtWidgets.QPushButton('Ok')
        self.btn.clicked.connect(self.grab)
        self.layout.addRow(self.btn)
        
        self.Label4 = QtWidgets.QLabel("Train")
        self.layout.addRow(self.Label4)
        
        self.graphicview = QtWidgets.QGraphicsView()
        self.layout.addRow(self.graphicview)
        
        self.Label7 = QtWidgets.QLabel("Test accuracy: ")
        self.layout.addRow(self.Label7)
        
        self.Label5 = QtWidgets.QLabel("Test")
        self.layout.addRow(self.Label5)
        
        self.graphicview2 = QtWidgets.QGraphicsView()
        self.layout.addRow(self.graphicview2)

        self.Label6 = QtWidgets.QLabel("Train accuracy: ")
        self.layout.addRow(self.Label6)
        
        self.Label8 = QtWidgets.QLabel("Weight: ")
        self.layout.addRow(self.Label8)
        
        self.setLayout(self.layout)
        self.setWindowTitle("HW1")
        self.setGeometry(150, 150, 400, 800)
        
    def grab(self):             #多線程處理
        print("Get process!")
        self.fileName = self.tmp1.text()
        self.rate = float(self.tmp2.text())
        self.round = int(self.tmp3.text())
        self.arrange = False
        self.count += 1
        self.check()
        
    def check(self):
        if self.rate < 0 or self.round < 0:
            print("Invalid input")
            return None

        train = Train()
        train.set(self.fileName, self.rate, self.round, self.count)
        self.layout.addRow(train.progress_bar)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene2 = QtWidgets.QGraphicsScene()
        train.run()
        graphicscene.addWidget(train.get_train_pic())
        graphicscene2.addWidget(train.get_test_pic())
        self.graphicview.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicview2.setScene(graphicscene2)
        self.graphicview.show()
        self.graphicview2.show()
        self.Label7.setText("Train accuracy: " + train.train_acc_text + "%")
        self.Label6.setText("Test accuracy: " + train.test_acc_text + "%")
        self.Label8.setText("Weight: " + train.weight)

def sgn(x):
    if x >= 0 :
        return 2
    else :
        return 1
    
def sgn_bin(x):
    if x >= 0 :
        return 1
    else :
        return 0

def main():
    
    app = QtWidgets.QApplication(sys.argv)
    window = Input()
    window.show()
     
    #close window 
    app.exec_() 
    
if __name__ == "__main__":  
    main()