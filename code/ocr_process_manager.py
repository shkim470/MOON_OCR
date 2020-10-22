import os 
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import threading
import socket
import subprocess
from queue import Queue
import pickle

count = 1
class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        self.statusBar()
        self.statusBar().showMessage('process manager started..')

        self.ocrlayout = OCRLayout(self)
        self.setCentralWidget(self.ocrlayout)

        self.setWindowTitle("OCR Process Manager")
        #self.setGeometry(50, 50, 1024, 768)
        self.show()

class OCRLayout(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)         
        

        self.proc_dict = {}
        #widget
        #self.ocr_folder_button = QPushButton('OCR Folder Path')
        #self.ocr_folder_button.clicked.connect(self.btn_clicked_folder_button)
        self.folder_path = os.getcwd()
        #self.view_ocr_path_label = QLabel(self)

        #self.set_config_button = QPushButton('설정')
        #self.set_config_button.clicked.connect(self.setConfig)

        #self.set_input_button = QPushButton('INPUT FOLER')
        #self.set_input_button.clicked.connect(self.btn_clicked_folder_button)
        #self.input_folder_path = ''
        #self.view_input_path_label = QLabel(self)


        #self.set_output_button = QPushButton('OUTPUT FOLER')
        #self.set_output_button.clicked.connect(self.btn_clicked_folder_button)
        #self.output_folder_path = ''
        #self.view_output_path_label = QLabel(self)

        

        self.add_button = QPushButton('프로세스 추가')
        self.add_button.clicked.connect(self.addProcess)

        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.initWidget()

        self.subprocess_id = 1
        self.list_subprocess_id=[]

        self.q = Queue()

        #server
        self.server = Thread_test(self.q)

        #update thread
        self.th = Worker(parent=self)
        self.th.sec_changed.connect(self.update)

        self.th.start()
        self.th.working=True

        #subprocess
        self.listp=[]

    @pyqtSlot(str)
    def update(self, msg):
        if self.q.qsize() != 0:
            while True:
                data = self.q.get()
                #print(data)
                items = self.grid.findChildren(QHBoxLayout)
                for item in items:
                    #print(item.count())
                    if(item.count() == 4):                    
                        for column in range(item.count()):
                            it = item.itemAt(column)
                            if it is not None:
                                widget = it.widget()
                                if isinstance(widget, QLineEdit) and widget.objectName() == str(int(data[0])):
                                    widget.setText(str(data[1]))
                if self.q.qsize() == 0:
                    break

    def initWidget(self):

        '''
        hlayout_ocr_folder_path = QHBoxLayout()
        self.view_ocr_path_label.setText(self.folder_path)            
        hlayout_ocr_folder_path.addWidget(self.ocr_folder_button, alignment=Qt.AlignLeft)        
        hlayout_ocr_folder_path.addWidget(self.view_ocr_path_label, alignment=Qt.AlignLeft)

        hlayout_input_folder_path = QHBoxLayout()
        self.view_input_path_label.setText('-')           
        hlayout_input_folder_path.addWidget(self.set_input_button, alignment=Qt.AlignLeft)        
        hlayout_input_folder_path.addWidget(self.view_input_path_label, alignment=Qt.AlignLeft)

        hlayout_output_folder_path = QHBoxLayout()
        self.view_output_path_label.setText('-')           
        hlayout_output_folder_path.addWidget(self.set_output_button, alignment=Qt.AlignLeft)        
        hlayout_output_folder_path.addWidget(self.view_output_path_label, alignment=Qt.AlignLeft)
        '''
        

        vlayout = QVBoxLayout()        
        #vlayout.addLayout(hlayout_ocr_folder_path)
        #vlayout.addLayout(hlayout_input_folder_path)
        #vlayout.addLayout(hlayout_output_folder_path)
        #vlayout.addStretch()
        #vlayout.addWidget(label, alignment=Qt.AlignTop)
        vlayout.addWidget(self.add_button, alignment=Qt.AlignTop)

        self.grid.addLayout(vlayout, 0, 0)

    def btn_clicked_folder_button(self):
        self.folder_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.view_ocr_path_label.setText(':'+self.folder_path)

    def addProcess(self):
        global count

        sub_process = QHBoxLayout()

        label = QLabel(str(count),self)

        #b_configFile = QPushButton('set Config', self)
        #b_configFile.setObjectName(str(count))
        #b_configFile.clicked.connect(self.setConfig)

        b_start = QPushButton('시작',self)
        b_start.setObjectName(str(count))
        b_start.clicked.connect(self.startButton)

        b_stop = QPushButton('중지', self)
        b_stop.setObjectName(str(count))
        b_stop.clicked.connect(self.stopButton)

        status = QLineEdit('Ready', self)
        status.setMinimumWidth(550)
        status.setMinimumHeight(10)
        status.setObjectName(str(count))

        sub_process.addWidget(label, alignment=Qt.AlignLeft)
        #sub_process.addWidget(b_configFile, alignment=Qt.AlignLeft)
        sub_process.addWidget(b_start, alignment=Qt.AlignLeft)
        sub_process.addWidget(b_stop, alignment=Qt.AlignLeft)
        sub_process.addWidget(status, alignment=Qt.AlignLeft)

        self.grid.addLayout(sub_process, count, 0)
        self.list_subprocess_id.append(count)
        count += 1

    def startButton(self):
        sender = self.sender()
        #folder_path = self.view_ocr_path_label.Text()
        print(self.folder_path)
        if self.folder_path != '':
            exe_path = (self.folder_path+'\\moonocr_proc\\moonocr_proc.exe')
            p = subprocess.Popen(exe_path, stdout=subprocess.PIPE, shell=True)
            #ctypes.windll.shell32.ShellExecuteEx(0,'open',exe_path,None,None,1)
        print('Start a procss...PID: '+sender.objectName())
        self.server.setSubprocessId(sender.objectName())
        self.proc_dict[sender.objectName()] = p.pid


    def stopButton(self):
        sender = self.sender()
        hosts = self.server.getHosts()
        for host in hosts:
            if sender.objectName() == str(host[2]):
                #address = host[:2]
                print ('Quit', host[0], host[1], host[2])
                self.server.setSubprocessId(sender.objectName())
                self.server.sendMessage('Quit', host[0], host[1])

        #self.server.mysocket
        print('Stop process..'+sender.objectName())

    def setConfig(self):
        sender = self.sender()
        print('config'+sender.objectName())

        if self.folder_path != '':
            file_path = ["notepad", self.folder_path+'\\process.conf']
            subprocess.Popen(file_path)


        # popup = PopupWindow()
        # popup.setWindowTitle('modal dialog')
        # popup.setWindowModality(Qt.NonModal)
        # popup.setWindowModality(Qt.ApplicationModal)
        # popup.exec_()

#        print(sender.text())

class PopupWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setModal(False)

    def paintEvent(self, e):
        dc = QStylePainter(self)
        dc.drawLine(0, 0, 100, 100)
        dc.drawLine(100, 0, 0, 100)

class Thread_test():
    def __init__(self, qu = None):
        HOST = "localhost"
        PORT = 4000

        self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)

        self.mysocket.bind((HOST, PORT))
        self.hosts = []        
        self.host_duplicate=[]

        #self.interval = interval
        self.packet = ''
        self.address = None
        self.subprocess_id = -1
        self.list_subprocess=[]
        self.list_port = []

        self.thread = threading.Thread(target=self.run, args=(qu, ))
        self.thread.daemon=True
        self.thread.start()

    def setSubprocessId(self, id=-1):
        self.subprocess_id = id
        if self.subprocess_id not in self.list_subprocess:
            self.list_subprocess.append(self.subprocess_id)

    def getHosts(self):
        return self.hosts

    def sendMessage(self, message=None, host = 'localhost', port = 4000):
        if message is not None:
            self.mysocket.sendto(pickle.dumps(message), (host, port))

    def run(self, q):
        while True:
            #print ("Listening...\n")
            packet, address = self.mysocket.recvfrom(1024)
            address2 = address + tuple(self.subprocess_id)
            duplicate = (address[1], int(self.subprocess_id))
            #address.extend(self.subprocess_id)
            #print('address count = '+str(self.hosts.count(address2)))
            if not(self.hosts.count(address2)) and not(self.list_port.count(address2[1])) :
                # for host in self.hosts:
                #     if host[1] != address2[1]:
                self.hosts.append(address2)
                self.list_port.append(address2[1])                

            # elif self.hosts.count(address2)
            #print("Connected sub-prosesses :", self.list_subprocess)            
            #print("Connected PID :", self.subprocess_id)
            #print("connect hosts :", self.hosts)
            #print("Packet received")
            #print("from : %s, port : %s"%(address2[0], address2[1]))
            #print("length :", len(packet))
            #print("content:", packet)

            recv_data = pickle.loads(packet)
            if recv_data == "quit":
                self.mysocket.sendto(pickle.dumps("quit"), address)
                self.hosts.remove(address2)
                self.list_port.remove(address2[1])
                #if not(len(self.hosts)):
                #    break
            elif recv_data == 'hi':
                self.mysocket.sendto(str(self.subprocess_id).encode(), address)
            else:
                #print(recv_data)
                q.put(recv_data)
        self.mysocket.close()

class Worker(QThread):
    sec_changed = pyqtSignal(str)

    def __init__(self, sec=0, parent=None):
        super().__init__()
        self.main = parent
        self.working = True
        self.sec = sec

    def __del__(self):
        print("... end thread......")
        self.wait()

    def run(self):
        while self.working:
            self.sec_changed.emit('time (secs) : {}'.format(self.sec))
            self.sleep(1)
            self.sec += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()

    sys.exit(app.exec_())


# import sys
# from PyQt5 import QtWidgets
# # Clas Mainwindow
# class Window(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#     def init_ui(self):
# #        createb(self)
#         self.btnClick = createb(self)     # +
#         self.show()
# # class that should create the buttons with click action so that when I create
# # the class the button is created with click action on the main window
# class createb():
#     def __init__(self, mainwindow):
#         self.mainwindow = mainwindow                       # <--- +
#         mainwindow.b = QtWidgets.QPushButton('Push Me')
#         mainwindow.l = QtWidgets.QLabel('I have not been clicked yet')
#         h_box = QtWidgets.QHBoxLayout()
#         h_box.addStretch()
#         h_box.addWidget(mainwindow.l)
#         h_box.addStretch()
#         v_box = QtWidgets.QVBoxLayout()
#         v_box.addWidget(mainwindow.b)
#         v_box.addLayout(h_box)
#         mainwindow.setLayout(v_box)
#         mainwindow.setWindowTitle('PyQt5 Lesson 5')
#         mainwindow.b.clicked.connect(self.btn_click)
#     def btn_click(self):
# #            self.l.setText('I have been clicked')
#             self.mainwindow.l.setText('I have been clicked')    # +
# app = QtWidgets.QApplication(sys.argv)
# a_window = Window()
# sys.exit(app.exec_())

