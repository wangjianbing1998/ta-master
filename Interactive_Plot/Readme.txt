pip install finplot==0.3.6  Release at Oct,2019，还算比价新了
就不要安装最新的finplot==0.8.0了，因为这个会导入PyQt5==5.13.0，这个包会自动导入PyQt5-sip==0.12.8，这个会与sip冲突
导致
from PyQt5 import QtGui, QtCore, QtWidgets, uic
这个找不到PyQt5.sip
这个bug耗费了我一个上午一个一个version试出来的