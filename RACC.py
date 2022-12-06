'''
Regression Adjusted Colocalization Color mapping (RACC)
When you use this software, please cite the following paper

Regression adjusted colocalization colour mapping (RACC): A novel visualization 
model for qualitative colocalization analysis of 3D fluorescence micrographs

Install Dependancies:
    pip install numpy
    pip install matplotlib
    pip install scikit-image
    pip install scipy
    pip install mayavi
    pip install PyQt5

Author: Rensu P. Theart (Stellenbosch University)
Project supervisors: Ben Loos, Thomas R. Niesler (Stellenbosch University)

Date: 26 March 2020
Version 0.81

Updates:
-fixed "skimage.data" has no atrribute "imread"
'''

import numpy as np
import math

import warnings
from enum import Enum

from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor
from tvtk.util import ctf

from skimage import data, io

#import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets
if int(QtCore.qVersion().split('.')[0]) == 5:
    from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


from PyQt5.QtCore import (Qt, QThread, QObject, pyqtSignal, pyqtSlot)
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, 
        QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton,  QSlider, QStyleFactory, QWidget,
        QFileDialog, QMessageBox, QVBoxLayout)

from traits.api import HasTraits, Instance, on_trait_change, Range
from traitsui.api import View, Item, HGroup


#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
class Worker(QObject):
    finished = pyqtSignal()
    processThread = pyqtSignal()


    @pyqtSlot()
    def procCounter(self): # A slot takes no params
#        for i in range(1, 100):
#            time.sleep(1)
        self.processThread.emit()

        self.finished.emit()

################################################################################
#The actual visualization
class VisualizationInput(HasTraits):
    scene = Instance(MlabSceneModel, ())
    scene.background = (0,0,0) # np.array([0.0,0.0,0.0])
    alpha = Range(0.0, 0.2, 0.1)
    scale = Range(0.0, 20, 10)
    fullStack3D = None
    RGBcolorm = []
    maxVal = 0    
    vol = None

    @on_trait_change('fullStack3D, scene.activated, RGBcolorm')#'scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
#        self.scene.mlab.test_points3d()
        if(len(self.RGBcolorm) == 0):
            self.RGBcolorm = self.generateColormap()
            self.maxVal = len(self.RGBcolorm)
        
        try:
            print ("\nself.fullStack3D.shape ", self.fullStack3D.shape)
            
#            mlab.figure(bgcolor = (0,0,0), fgcolor = (1, 1, 1))
            
            self.vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.fullStack3D) )#, vmin=0.0, vmax=0.5)
            self.vol.volume.scale = np.array([self.scale, 1.0, 1.0])
            self.vol._volume_property.shade = False
            self.vol._volume_property.interpolation_type = 'nearest'
            #vol._volume_mapper = tvtk.FixedPointVolumeRayCastMapper()
            #vol._volume_mapper.on_trait_change(vol.render)
            self.vol.render()
            
            c = ctf.save_ctfs(self.vol._volume_property)
            #print( c)
            c['range'] = (0, self.maxVal)
            c['alpha'] = [[0.0, 0.0], [1.0, self.alpha], [self.maxVal, self.alpha]]#[[0.0, 0.1], [1.0, 0.1], [maxVal-1, 0.1], [maxVal, 0.1]]
            c['rgb'] = self.RGBcolorm#[[0.0, 0.0, 0.0, 0.0], [maxVal/4, 1.0, 0.0, 0.0], [maxVal/2, 1.0, 1.0, 0.0], [maxVal/4*3, 0.0, 1.0, 0.0], [maxVal, 0.0, 0.0, 0.0]]
            #print( c)
            #c['rgb'] = 
            ctf.load_ctfs(c, self.vol._volume_property)
            # Update the shadow LUT of the volume module.
            self.vol.update_ctf = True
            
#            mlab.view(132, 54, 45, [21, 20, 21.5])
#            mlab.show()
            self.scene.background = (0,0,0)
            
            print ("3D stack viz")
        except:
            print ("\nNo stack yet")
            
    @on_trait_change('scale')#'scene.activated')
    def update_plot2(self):
        self.vol.volume.scale = np.array([self.scale, 1.0, 1.0])
        
    @on_trait_change('alpha')#'scene.activated')
    def update_plot3(self):
        c = ctf.save_ctfs(self.vol._volume_property)
        #print( c)
#        c['range'] = (0, self.maxVal)
        c['alpha'] = [[0.0, 0.0], [1.0, self.alpha], [self.maxVal, self.alpha]]#[[0.0, 0.1], [1.0, 0.1], [maxVal-1, 0.1], [maxVal, 0.1]]
#        c['rgb'] = self.RGBcolorm#[[0.0, 0.0, 0.0, 0.0], [maxVal/4, 1.0, 0.0, 0.0], [maxVal/2, 1.0, 1.0, 0.0], [maxVal/4*3, 0.0, 1.0, 0.0], [maxVal, 0.0, 0.0, 0.0]]
        #print( c)
        #c['rgb'] = 
        ctf.load_ctfs(c, self.vol._volume_property)
        # Update the shadow LUT of the volume module.
        self.vol.update_ctf = True
            
    def generateColormap(self, bitsPerChannel=6):
        maxVal = 2**bitsPerChannel
        colormap = []
        
        chan1Val = 0
        chan2Val = 0
        for i in range(0,maxVal*maxVal):
            if(chan1Val == maxVal):
                chan2Val += 1
                chan1Val = 0
                
            colormap.append([i, chan1Val/maxVal, chan2Val/maxVal, 0.0])
            chan1Val += 1
            
        return colormap

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=300, width=490, show_label=False), HGroup(
                        '_', 'alpha', 'scale',
                    ), resizable=True
    )

################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.visualization = VisualizationInput()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()
        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        self.layout.addWidget(self.ui)
        self.ui.setParent(self)
    
    def refresh(self):
        # The edit_traits call will generate the widget to embed.
        self.layout.removeWidget(self.ui)
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        self.layout.addWidget(self.ui)
        self.ui.setParent(self)

class OutputType(Enum):
    TIF = 0
    PNG = 1
    JPG = 2

class RACC(QDialog):
    def __init__(self, parent=None):
        super(RACC, self).__init__(parent)
        
        self.colormap = io.imread("magmaLine.png").astype(np.uint8)[0,:,0:3]
        self.colormap[0] = np.zeros(3)
    
        self.originalPalette = QApplication.palette()
    
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())
        styleName = "Fusion"
        index = styleComboBox.findText(styleName)
        if ( index != -1 ): # -1 for not found
           styleComboBox.setCurrentIndex(index)
        
    
        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)
    
        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(False)
        
        styleComboBox.activated[str].connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        self.createFileSelectGroupBox()
        self.createParameterGroupBox()
        self.createOutputSettingsGroup()

        processButton = QPushButton("Process")
        processButton.setDefault(True)
        processButton.clicked.connect(self.process)
        processButton.setFixedHeight(40)
        
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        
        self.mayavi_input3D = MayaviQWidget(self)
        self.mayavi_output3D = MayaviQWidget(self)        
        
        self.static_canvasInput2D = FigureCanvas(Figure(frameon=False, tight_layout=True))    
        #self.addToolBar(NavigationToolbar(self.mayavi_input3D, self))
        self._static_ax = self.static_canvasInput2D.figure.add_axes([0, 0, 1, 1])
        self._static_ax.axis('off')

        self.static_canvasOutput2D = FigureCanvas(Figure( frameon=False, tight_layout=True))    
        #self.addToolBar(NavigationToolbar(self.mayavi_input3D, self))
        self._static_axOutput = self.static_canvasOutput2D.figure.add_axes([0, 0, 1, 1])
        self._static_axOutput.axis('off')

        self.mainLayout = QGridLayout()
        self.mainLayout.addLayout(topLayout, 0, 0)
        self.mainLayout.addWidget(self.fileSelectGroupBox, 1, 0)
        self.mainLayout.addWidget(self.parameterGroupBox, 2, 0)
        self.mainLayout.addWidget(self.outputGroupBox, 3, 0)
        self.mainLayout.addWidget(processButton, 4,0)
        self.mainLayout.addWidget(self.progressBar, 5,0)
        self.mainLayout.addWidget(self.mayavi_input3D, 0,1,6,1)
        self.mainLayout.addWidget(self.mayavi_output3D, 0,2,6,1)
        self.mainLayout.addWidget(self.static_canvasInput2D, 0,1,6,1)
        self.mainLayout.addWidget(self.static_canvasOutput2D, 0,2,6,1)
        self.mainLayout.setColumnMinimumWidth(1,1)
        self.mainLayout.setColumnMinimumWidth(2,1)
        self.mainLayout.setColumnStretch(0,1)
        self.mainLayout.setColumnStretch(1,0)
        self.mainLayout.setColumnStretch(2,0)
        self.mayavi_input3D.hide()
        self.mayavi_output3D.hide()
        self.static_canvasInput2D.hide()
        self.static_canvasOutput2D.hide()
        
        self.setLayout(self.mainLayout)
        
        self.setWindowTitle("RACC")
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.setWindowFlags(QtCore.Qt.Window) 
        self.changeStyle(styleName)
        
    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()
        
    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)
    
    def selectFile1(self):
        returnedText = QFileDialog.getOpenFileName()[0]
        
        if(returnedText != ""):
            self.channel1_FileLE.setText(returnedText)    
        
    def selectFile2(self):
        returnedText = QFileDialog.getOpenFileName()[0]
        
        if(returnedText != ""):
            self.channel2_FileLE.setText(returnedText)    
        
    def threshSliderValueChanged1(self):
        self.ch1ThreshLE.setText(str(self.channel1_Thresh_slider.value()))
        
    def threshSliderValueChanged2(self):
        self.ch2ThreshLE.setText(str(self.channel2_Thresh_slider.value()))
        
    def thetaSliderValueChanged(self):
        self.thetaLE.setText(str(self.theta_slider.value()))        
        
    def percentageSliderValueChanged(self):
        self.percentageLE.setText(str(self.percentage_slider.value()))
        
    def theshLineEditChanged1(self):
        try:
            if int(self.ch1ThreshLE.text()) > 255:
                self.ch1ThreshLE.setText("255")
            elif int(self.ch1ThreshLE.text()) < 0:
                self.ch1ThreshLE.setText("0")
        except ValueError:
            self.ch1ThreshLE.setText("0")
            
        self.channel1_Thresh_slider.setValue(int(self.ch1ThreshLE.text()))
        
    def theshLineEditChanged2(self):
        try:
            if int(self.ch2ThreshLE.text()) > 255:
                self.ch2ThreshLE.setText("255")
            elif int(self.ch2ThreshLE.text()) < 0:
                self.ch2ThreshLE.setText("0")
        except ValueError:
            self.ch2ThreshLE.setText("0")
            
        self.channel2_Thresh_slider.setValue(int(self.ch2ThreshLE.text()))
            
        
    def thetaLineEditChanged(self):
        try:
            if int(self.thetaLE.text()) > 89:
                self.thetaLE.setText("89")
            elif int(self.thetaLE.text()) < 0:
                self.thetaLE.setText("0")
        except ValueError:
            self.thetaLE.setText("0")
            
        self.theta_slider.setValue(int(self.thetaLE.text()))
        
    def percentageLineEditChanged(self):
        try:
            if int(self.percentageLE.text()) > 100:
                self.percentageLE.setText("100")
            elif int(self.percentageLE.text()) < 0:
                self.percentageLE.setText("0")
        except ValueError:
            self.percentageLE.setText("0")
            
        self.percentage_slider.setValue(int(self.percentageLE.text()))
                
    def createFileSelectGroupBox(self):
        self.fileSelectGroupBox = QGroupBox("Input File Select")
        
        ch1_label = QLabel("Channel 1")
        self.channel1_FileLE = QLineEdit("Red2D.png") #"RedSphere.tif")
        self.channel1_FileLE.setToolTip("Used TIFF for 3D (RED channel)")
        ch1BrowseButton = QPushButton("Browse")
        #ch1BrowseButton.setDefault(True)
        ch1BrowseButton.clicked.connect(self.selectFile1)
        
        ch2_label = QLabel("Channel 2")
        self.channel2_FileLE = QLineEdit("Green2D.png")#"GreenSphere.tif")
        self.channel1_FileLE.setToolTip("Used TIFF for 3D (GREEN channel)")
        ch2BrowseButton = QPushButton("Browse")
        ch2BrowseButton.clicked.connect(self.selectFile2)
        
        self.visualizeInputFilesCheckBox = QCheckBox("&Visualize input")
        self.visualizeInputFilesCheckBox.setChecked(True)
                
        layout = QGridLayout()
        layout.addWidget(ch1_label,0,0)
        layout.addWidget(self.channel1_FileLE,0,1)
        layout.addWidget(ch1BrowseButton,0,2)
        
        layout.addWidget(ch2_label, 1,0)
        layout.addWidget(self.channel2_FileLE,1,1)
        layout.addWidget(ch2BrowseButton,1,2)
        layout.setColumnMinimumWidth(1,300)
        layout.addWidget(self.visualizeInputFilesCheckBox, 2,0,1,3)
        
        self.fileSelectGroupBox.setLayout(layout)
        
    def createParameterGroupBox(self):
        self.parameterGroupBox = QGroupBox("Parameters")
        
        defaultSliderVal = 5
        #Channel 1
        ch1_label = QLabel("Channel 1 Threshold")
        self.channel1_Thresh_slider = QSlider(Qt.Horizontal, self.parameterGroupBox)
        self.channel1_Thresh_slider.setTickPosition(QSlider.TicksBothSides)
        self.channel1_Thresh_slider.setTickInterval(32)
        self.channel1_Thresh_slider.setRange(0,255)
        self.channel1_Thresh_slider.setSingleStep(1)
        self.channel1_Thresh_slider.setValue(defaultSliderVal)
        
        labelMin1 = QLabel("0");
        labelMax1 = QLabel("255");
        self.ch1ThreshLE = QLineEdit(str(defaultSliderVal))
        self.ch1ThreshLE.setFixedWidth(40)
        self.ch1ThreshLE.textChanged.connect(self.theshLineEditChanged1)
        
        self.channel1_Thresh_slider.valueChanged.connect(self.threshSliderValueChanged1)
        
        layoutSlider1 = QGridLayout();
        layoutSlider1.addWidget(self.channel1_Thresh_slider, 0, 1 );
        layoutSlider1.addWidget(labelMin1, 0, 0);
        layoutSlider1.addWidget(labelMax1, 0, 2);
        layoutSlider1.addWidget(self.ch1ThreshLE, 0, 3);
    
        ########################################################
        
        #Channel 2
        ch2_label = QLabel("Channel 2 Threshold")
        self.channel2_Thresh_slider = QSlider(Qt.Horizontal, self.parameterGroupBox)
        self.channel2_Thresh_slider.setTickPosition(QSlider.TicksBothSides)
        self.channel2_Thresh_slider.setTickInterval(32)
        self.channel2_Thresh_slider.setRange(0,255)
        self.channel2_Thresh_slider.setSingleStep(1)
        self.channel2_Thresh_slider.setValue(defaultSliderVal)
        
        labelMin2 = QLabel("0");
        labelMax2 = QLabel("255");
        self.ch2ThreshLE = QLineEdit(str(defaultSliderVal))
        self.ch2ThreshLE.setFixedWidth(40)
        self.ch2ThreshLE.textChanged.connect(self.theshLineEditChanged2)
        
        self.channel2_Thresh_slider.valueChanged.connect(self.threshSliderValueChanged2)
        
        layoutSlider2 = QGridLayout();
        layoutSlider2.addWidget(self.channel2_Thresh_slider, 0, 1 );
        layoutSlider2.addWidget(labelMin2, 0, 0);
        layoutSlider2.addWidget(labelMax2, 0, 2);
        layoutSlider2.addWidget(self.ch2ThreshLE, 0, 3);
        
        ########################################################
        
        #Penelization factor
        theta_label = QLabel("Penelization factor (theta)")
        self.theta_slider = QSlider(Qt.Horizontal, self.parameterGroupBox)
        self.theta_slider.setTickPosition(QSlider.TicksBothSides)
        self.theta_slider.setTickInterval(15)
        self.theta_slider.setRange(0,89)
        self.theta_slider.setSingleStep(1)
        self.theta_slider.setValue(45)
        
        labelMinTheta = QLabel("0");
        labelMaxTheta= QLabel("89");
        self.thetaLE = QLineEdit(str(45))
        self.thetaLE.setFixedWidth(40)
        self.thetaLE.textChanged.connect(self.thetaLineEditChanged)
        
        self.theta_slider.valueChanged.connect(self.thetaSliderValueChanged)
        
        layoutSliderTheta = QGridLayout();
        layoutSliderTheta.addWidget(self.theta_slider, 0, 1 );
        layoutSliderTheta.addWidget(labelMinTheta, 0, 0);
        layoutSliderTheta.addWidget(labelMaxTheta, 0, 2);
        layoutSliderTheta.addWidget(self.thetaLE, 0, 3);        
        
        ########################################################
        
        #Percentage to include
        percentage_label = QLabel("Percentage to include")
        self.percentage_slider = QSlider(Qt.Horizontal, self.parameterGroupBox)
        self.percentage_slider.setTickPosition(QSlider.TicksBothSides)
        self.percentage_slider.setTickInterval(25)
        self.percentage_slider.setRange(0,100)
        self.percentage_slider.setSingleStep(1)
        self.percentage_slider.setValue(99)
        
        labelMinPercentage = QLabel("0");
        labelMaxPercentage= QLabel("100");
        self.percentageLE = QLineEdit(str(99))
        self.percentageLE.setFixedWidth(40)
        self.percentageLE.textChanged.connect(self.percentageLineEditChanged)
        
        self.percentage_slider.valueChanged.connect(self.percentageSliderValueChanged)
        
        layoutSliderPercentage = QGridLayout();
        layoutSliderPercentage.addWidget(self.percentage_slider, 0, 1 );
        layoutSliderPercentage.addWidget(labelMinPercentage, 0, 0);
        layoutSliderPercentage.addWidget(labelMaxPercentage, 0, 2);
        layoutSliderPercentage.addWidget(self.percentageLE, 0, 3);
        
        
                
        layout = QGridLayout()
        layout.addWidget(ch1_label,0,0)
        layout.addLayout(layoutSlider1,0,1)        
        
        layout.addWidget(ch2_label, 1,0)
        layout.addLayout(layoutSlider2,1,1)
        
        layout.addWidget(theta_label, 2,0)
        layout.addLayout(layoutSliderTheta,2,1)
        
        layout.addWidget(percentage_label, 3,0)
        layout.addLayout(layoutSliderPercentage,3,1)
        self.parameterGroupBox.setLayout(layout)    
        
    def createOutputSettingsGroup(self):
        self.outputGroupBox = QGroupBox("Output Settings")
        
        self.outputFileTypeComboBox = QComboBox()
        self.outputFileTypeComboBox.addItem("TIF stack")
        self.outputFileTypeComboBox.addItem("PNG slices")
        self.outputFileTypeComboBox.addItem("JPG slices")
            
        fileTypeLabel = QLabel("&Output File type:")
        fileTypeLabel.setBuddy(self.outputFileTypeComboBox)       
        
        self.grayscaleOutput = QCheckBox("&Output in grayscale")
        self.grayscaleOutput.setChecked(False)
        
        self.visualizeOutput = QCheckBox("&Visualize output")
        self.visualizeOutput.setChecked(True)
        
        outputTypeLayout = QHBoxLayout()
        outputTypeLayout.addWidget(fileTypeLabel)
        outputTypeLayout.addWidget(self.outputFileTypeComboBox)
        outputTypeLayout.addWidget(self.grayscaleOutput)
        outputTypeLayout.addStretch(1)
        outputTypeLayout.addWidget(self.visualizeOutput)
        
        self.outputGroupBox.setLayout(outputTypeLayout)
    
        
    def process(self):
        if(self.visualizeInputFilesCheckBox.isChecked()):
            self.mainLayout.setColumnMinimumWidth(1,490)
            self.mainLayout.setColumnStretch(1,2)
        else:
            self.mainLayout.setColumnMinimumWidth(1,1)
            self.mainLayout.setColumnStretch(1,0)
            
        if(self.visualizeOutput.isChecked()):
            self.mainLayout.setColumnMinimumWidth(2,490)
            self.mainLayout.setColumnStretch(2,2)
        else:
            self.mainLayout.setColumnMinimumWidth(2,1)
            self.mainLayout.setColumnStretch(2,0)
            
        self.mayavi_input3D.hide()
        self.mayavi_output3D.hide()
        self.static_canvasInput2D.hide()
        self.static_canvasOutput2D.hide()
        
        
        self.obj = Worker()  # no parent!
        self.thread = QThread()  # no parent!
        self.obj.processThread.connect(self.processThread)
        
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.procCounter)
        
        self.thread.start()
    
        
        
    def processThread(self):  
        # Preprocessing and setup
        threshCh1 = int(self.ch1ThreshLE.text())
        threshCh2 = int(self.ch2ThreshLE.text())
        
        penFactor = int(self.thetaLE.text())
        percToInclude = int(self.percentageLE.text())/100.0
        
        print("Process with values:\nThres Ch1: {}\nThres Ch2: {}".format(threshCh1, threshCh2))
        print("Penelization factor (theta): {}\nPercentage to include: {}".format(penFactor, percToInclude*100))
        
        # contains values 0-255
        ch1Stack = io.imread(self.channel1_FileLE.text())
        ch2Stack = io.imread(self.channel2_FileLE.text())    
        
        maxIntensity1 = np.max(ch1Stack)
        maxIntensity2 = np.max(ch2Stack)
        maxInt = np.max((maxIntensity1, maxIntensity2))
        
        if(maxInt > 255):
            print("Max intensity greater than 255 ({}), adjusted".format(maxInt))
            ch1Stack = ch1Stack / maxInt *255
            ch2Stack = ch2Stack / maxInt *255
        
        
        #originalShape = ch1Stack.shape
        
        print("\nch1Stack.shape", ch1Stack.shape)
        print("ch2Stack.shape", ch2Stack.shape)    

        self.progressBar.setValue(5)
                
        isStack = True #3D stack if True, otherwise single slice 2D image
        if(ch1Stack.shape != ch2Stack.shape):
            print("\nERROR: stack shapes are not the same, cannot continue.")
            self.showErrorDialog("Stack shapes are not the same, cannot continue.\n{} vs {}".format(ch1Stack.shape, ch2Stack.shape))
            return
        
        # "flatten" channels, if single channel image, as is supposed, this will have the desired effect
        if(len(ch1Stack.shape) == 4): # assuming (slices,x,y,RGB/RGBA)
            # remove alpha channel if exists
            ch1Stack = ch1Stack[:,:,:,0:3]
            ch2Stack = ch2Stack[:,:,:,0:3]
            
            originalShape = ch1Stack.shape

            ch1Stack = np.amax(ch1Stack, axis=3)
            ch2Stack = np.amax(ch2Stack, axis=3)

            #visualization            
            if(self.visualizeInputFilesCheckBox.isChecked()):
                self.mayavi_input3D.show()                
                
                self.mayavi_input3D.visualization = VisualizationInput()
                self.mayavi_input3D.visualization.fullStack3D = ch1Stack/4 + (np.ceil(ch2Stack/4))*64
            
                self.mayavi_input3D.refresh()
            
            isStack = True
            
            print("\nExtracted only intensity values")
            print("ch1Stack.shape", ch1Stack.shape)
            print("ch2Stack.shape", ch2Stack.shape)
            

        elif(len(ch1Stack.shape) == 3): # assuming (x, y, RGB/RGBA)
            # remove alpha channel if exists
            if ch1Stack.shape[-1] == 3 and ch1Stack.shape[-1] != ch1Stack.shape[-2]:
                ch1Stack = ch1Stack[:,:,0:3]
                ch2Stack = ch2Stack[:,:,0:3]

                #visualization
                if(self.visualizeInputFilesCheckBox.isChecked()):
                    #self.mainLayout.setColumnMinimumWidth(1,490)
                    self.static_canvasInput2D.show()
                    self._static_ax.clear()
                    self._static_ax.imshow(ch1Stack + ch2Stack)
                    self._static_ax.axis('off')
                    self._static_ax.figure.canvas.draw()

                originalShape = ch1Stack.shape

                ch1Stack = np.amax(ch1Stack, axis=2)
                ch2Stack = np.amax(ch2Stack, axis=2)

                isStack = False

                print("\nExtracted only intensity values")
                print("ch1Stack.shape", ch1Stack.shape)
                print("ch2Stack.shape", ch2Stack.shape)
            else:

                originalShape = tuple(list(ch1Stack.shape) + [3])

                # visualization
                if (self.visualizeInputFilesCheckBox.isChecked()):
                    self.mayavi_input3D.show()

                    self.mayavi_input3D.visualization = VisualizationInput()
                    self.mayavi_input3D.visualization.fullStack3D = ch1Stack / 4 + (np.ceil(ch2Stack / 4)) * 64

                    self.mayavi_input3D.refresh()

                isStack = True

                print("\nExtracted only intensity values")
                print("ch1Stack.shape", ch1Stack.shape)
                print("ch2Stack.shape", ch2Stack.shape)
            
            
        self.progressBar.setValue(10)
        #################################
        # Calculate RACC
        
        print("\n\nSTART of RACC calculation:")
        
        theta = penFactor * math.pi / 180.0;
        dThresh = 255
        xMax = -1
        yMax = -1
        
        Imax = 255
        texSize = 256
        
       
        #####################
        # CALCULATE averages and covariances
        
        valuesAboveThreshCh1 = ch1Stack[np.where(ch1Stack >= threshCh1)]
        valuesAboveThreshCh2 = ch2Stack[np.where(ch2Stack >= threshCh2)]
        		
        averageCh1 = np.average(valuesAboveThreshCh1)
        averageCh2 = np.average(valuesAboveThreshCh2)
        
        print("\nAverage Ch1: {}\nAverage Ch2: {}".format(averageCh1, averageCh2))
        		
        filteredCh1Stack = np.copy(ch1Stack)
        filteredCh2Stack = np.copy(ch2Stack)
        filteredCh1Stack[filteredCh1Stack < threshCh1] = 0
        filteredCh2Stack[filteredCh2Stack < threshCh2] = 0
        
        filteredCh1Stack = filteredCh1Stack.ravel()
        filteredCh2Stack = filteredCh2Stack.ravel()
        
        covariance = np.cov(filteredCh1Stack, filteredCh2Stack)
        varXX = covariance[0,0]
        varYY = covariance[1,1]
        varXY = covariance[0,1]
        
        print("\nCovariance(xx): {}\nCovariance(yy): {}\nCovariance(xy): {}".format(varXX, varYY, varXY))
        self.progressBar.setValue(20)
        
        #####################
        # CALCULATE B0 and B1
        
        lamb = 1 #special case of Deming regression
        val = lamb * varYY - varXX
        
        B0 = 0
        B1 = 0
        
        if( varXY < 0):
            print("\nThe covariance is negative")
            B1 = (val - math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)
        else:
            B1 = (val + math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)
        
        B0 = averageCh2 - B1 * averageCh1
        
        print("\nB0 = {}  B1 = {}".format(B0, B1 ))
        
        self.progressBar.setValue(25)
        
        #####################
        # CALCULATE p0 and p1
        
        p0 = np.zeros(2)
        p1 = np.zeros(2)
        
        # For P0
        if(threshCh2<=threshCh1*B1 + B0):
        	p0[0] = threshCh1
        	p0[1] = threshCh1*B1 + B0
        elif(threshCh2>threshCh1*B1 + B0):
        	p0[0] = (threshCh2 - B0)/B1
        	p0[1] = threshCh2
        
        # For P1
        if(B0>= Imax*(1-B1)):
        	p1[0] = (Imax - B0)/B1
        	p1[1] = Imax
        elif(B0< Imax*(1-B1)):
        	p1[0] = Imax
        	p1[1] = Imax*B1 + B0
            
        print("\nP0 = {}  P1 = {}".format(p0, p1 ))
        
        self.progressBar.setValue(30)
        
        #####################
        # CALCULATE xMax
        
        totalVoxelCount = 0        
        colorMapFrequencyX = np.zeros(texSize)
        colorMapFrequencyY = np.zeros(texSize)
        
        overlappingSection = np.multiply(np.clip(filteredCh1Stack, 0, 1), np.clip(filteredCh2Stack, 0, 1))*Imax
        reducedCh1Stack = filteredCh1Stack[overlappingSection > 0]
        reducedCh2Stack = filteredCh2Stack[overlappingSection > 0]
        print("\nFull size was {} reduced colocalized size is {}. Remaining percentage {}%".format(filteredCh1Stack.shape, reducedCh1Stack.shape, reducedCh1Stack.shape[0]/filteredCh1Stack.shape[0]*100))
        
        
        totalVoxelCount = reducedCh2Stack.shape[0]
        print("Total Voxel Count: ", totalVoxelCount)
        qMat = np.stack((reducedCh1Stack, reducedCh2Stack))
        
        kMat = ((p1[1] - p0[1]) * (qMat[0] - p0[0]) - (p1[0] - p0[0]) * (qMat[1] - p0[1])) / ((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) *(p1[0] - p0[0]))
        
        xMat = qMat[0] - kMat * (p1[1] - p0[1])
        yMat = qMat[1] + kMat * (p1[0] - p0[0])
        
        fracXMat = (xMat-p0[0])/(p1[0]-p0[0])
        fracXMat = (np.clip(fracXMat, 0, 1)*Imax).astype(int)
        unique, counts = np.unique(fracXMat, return_counts=True)
        
        if(len(unique) >= 256):
            print("Note (this should never happen): Had to crop values to 255, max was ", len(unique))
            unique = unique[0:255]
            counts = counts[0:255]
        colorMapFrequencyX[unique] = counts
        
        fracYMat = (yMat-p0[1])/(p1[1]-p0[1])
        fracYMat = (np.clip(fracYMat, 0, 1)*Imax).astype(int)
        unique, counts = np.unique(fracYMat, return_counts=True)
        if(len(unique) >= 256):
            print("Note (this should never happen): Had to crop values to 255, max was ", len(unique))
            unique = unique[0:255]
            counts = counts[0:255]
        colorMapFrequencyY[unique] = counts
        
         # iterative implementation (MUCH slower)       
#        for i in range(0,reducedCh1Stack.shape[0]):
#            val_1 = reducedCh1Stack[i]
#            val_2 = reducedCh2Stack[i]            
#            
#            #if val_1 > 0 and val_2 > 0: #should always be true
#            totalVoxelCount += 1
#            
#            qi = np.array([val_1, val_2])
#            
#            k = ((p1[1] - p0[1]) * (qi[0] - p0[0]) - (p1[0] - p0[0]) * (qi[1] - p0[1])) / ((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) *(p1[0] - p0[0]))
#            xi = qi[0] - k * (p1[1] - p0[1])
#            yi = qi[1] + k * (p1[0] - p0[0])
#            
#            fracX = (xi-p0[0])/(p1[0]-p0[0])
#            fracX = np.clip(fracX, 0, 1)
#            colorMapFrequencyX[int(fracX*Imax)] += 1
#            
#            fracY = (yi-p0[1])/(p1[1]-p0[1])
#            fracY = np.clip(fracY, 0, 1)
#            colorMapFrequencyY[int(fracY*Imax)] += 1
#                
        self.progressBar.setValue(40)
        
        cumulativeTotalX = 0
        cumulativeTotalY = 0
        
        for i in range(0, texSize):
            cumulativeTotalX += colorMapFrequencyX[i]
            cumulativeTotalY += colorMapFrequencyY[i]
            
            if(cumulativeTotalX/totalVoxelCount >= percToInclude):
                xMax = (p1[0] - p0[0]) * (i/texSize) + p0[0]
                break      	
            elif(cumulativeTotalY/totalVoxelCount >= percToInclude):
                yMax = (p1[1] - p0[1]) * (i/texSize) + p0[1]
                break
        
        
        # some verification code
        if(xMax < 0 and yMax < 0):
        	if(B1 < 1):
        		xMax = Imax
        	else:
        		yMax = Imax
        
        print("\nMax X: {} / {} ({}%)".format(cumulativeTotalX, totalVoxelCount, cumulativeTotalX/totalVoxelCount*100));
        print("Max Y: {} / {} ({}%)".format(cumulativeTotalY, totalVoxelCount, cumulativeTotalY/totalVoxelCount*100));
        print("X_Max: {}  Y_Max: {}".format(xMax, yMax));
        self.progressBar.setValue(50)
        
        #####################
        # CALCULATE distance threshold (variant of vinary search)
        
        distanceCount = 0
        dThresh = 0
        tryCount = 0
        dMin = 0
        dMax = Imax
        
        print("\nCalc Distance iteration: ", end="")
        while(tryCount <= Imax):        
            tryCount += 1
            print("{}  ".format(tryCount), end="")
            
            dThresh = dMin + (dMax - dMin)/2.0        	
            
            dMat = ((abs((p1[1] - p0[1]) * qMat[0] - (p1[0] - p0[0]) * qMat[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])))
            distanceCount = np.where(dMat < dThresh)[0].shape[0]

            #iterative approach (MUCH slower)
#            distanceCount = 0
#            for i in range(0,reducedCh1Stack.shape[0]):
#                val_1 = reducedCh1Stack[i]
#                val_2 = reducedCh2Stack[i]
#                
#                #if val_1 > 0 and val_2 > 0: #should always be true
#                qi = np.array([val_1, val_2])
#                
#                #https://stackoverflow.com/questions/1811549/perpendicular-on-a-line-from-a-given-point/1811636#1811636
#                k = ((p1[1] - p0[1]) * (qi[0] - p0[0]) - (p1[0] - p0[0]) * (qi[1] - p0[1])) / ((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) *(p1[0] - p0[0]))
#                xi = qi[0] - k * (p1[1] - p0[1])
#                
#                d = ((abs((p1[1] - p0[1]) * qi[0] - (p1[0] - p0[0]) * qi[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])));
#                
#                if (d < dThresh):
#                    distanceCount += 1
            
            if( distanceCount/totalVoxelCount == percToInclude):
                break
            elif ( distanceCount/totalVoxelCount < percToInclude ):
                dMin = int(dThresh) + 1
            else:
                dMax = int(dThresh) - 1;
                
            if (dMin == dMax):
                break
            
            self.progressBar.setValue(self.progressBar.value() + 1)

        
        print("\n\nDistance threshold for {}% = {} (within {} times)".format(percToInclude*100,dThresh, tryCount ));
        
        if(xMax != -1):
            p1[0] = xMax;
            p1[1] = B1*p1[0] + B0;
        else:
            p1[1] = yMax;
            p1[0] = (p1[1] - B0)/B1;
        
        print("p_max for {}% = {}".format(percToInclude*100, p1));
        
        self.progressBar.setValue(60)
        
        #####################
        # Generate Ci greyscale map
        p0 = p0/Imax
        p1 = p1/Imax
        
        filteredCh1Stack[overlappingSection == 0] = 0
        filteredCh2Stack[overlappingSection == 0] = 0
        self.progressBar.setValue(61)
        
        output = np.zeros_like(filteredCh1Stack)
        qMat = np.stack((filteredCh1Stack, filteredCh2Stack)) /Imax
        self.progressBar.setValue(62)
        kMat = ((p1[1] - p0[1]) * (qMat[0] - p0[0]) - (p1[0] - p0[0]) * (qMat[1] - p0[1])) / ((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) *(p1[0] - p0[0]))        
        self.progressBar.setValue(63)
        xMat = qMat[0] - kMat * (p1[1] - p0[1])
        self.progressBar.setValue(64)
        dMat = ((abs((p1[1] - p0[1]) * qMat[0] - (p1[0] - p0[0]) * qMat[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])))
        
        self.progressBar.setValue(65)
        
        condition = np.logical_and((dMat*(p1[0]-p0[0])*math.tan(theta) + p0[0] < xMat), (xMat < p1[0]))
        output[condition] = ((xMat[condition]-p0[0])/(p1[0]-p0[0]) - dMat[condition]*math.tan(theta))*Imax
        self.progressBar.setValue(66)
        condition = xMat >= p1[0]
        output[condition] = np.clip(((1 - dMat[condition]*math.tan(theta))*Imax), 0,255)
        self.progressBar.setValue(68)
        condition = np.logical_or((xMat <= dMat*(p1[0]-p0[0])*math.tan(theta) + p0[0]), (dMat > dThresh/Imax))
        output[condition] = 0
        
        self.progressBar.setValue(70)
        
        print("\nFINISHED processing")
        
        
        if(self.grayscaleOutput.isChecked()):
            print("\nOutput in Grayscale")
            output = output.reshape(ch1Stack.shape)
            grayColormap = output
        else:
            print("\nOutput in Color")
            grayColormap = output.reshape(ch1Stack.shape)
            output = self.colormap[output]
            output = output.reshape(originalShape)
        
        print("Output shape: ", output.shape)
        self.progressBar.setValue(80)       

        
        outputType = self.outputFileTypeComboBox.currentIndex()
        
        if(outputType == OutputType.TIF.value):
            fileName = ""
            if isStack:
                fileName = self.saveFileDialog("3D Image Stack (*.TIFF)")
            else:
                fileName = self.saveFileDialog("2D image (*.TIFF)")
            
            if fileName == "" or fileName == None:
                fileName = "output.tif"
            elif not (fileName.lower().endswith("tiff") or fileName.lower().endswith("tif")):
                fileName += ".tif"
            io.imsave("{}".format(fileName), output)
            print("\nFINISHED writing to '{}'".format(fileName))                


        if(outputType == OutputType.PNG.value or outputType == OutputType.JPG.value):            
            if isStack:
                fileName = self.saveFileDialog("Image Stack (separate slices) (*.{})".format( OutputType(outputType).name))
                fileName = fileName.replace("." + OutputType(outputType).name, "")
                if fileName == "" or fileName == None:
                    fileName = "output"
                    
                    
                stackSize = output.shape[0]
                zeroPad = ":0"+str(len(str(stackSize)))+"d"
                
                for i in range(0,stackSize):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        io.imsave(str("{}{"+zeroPad+"}.{}").format(fileName, i, OutputType(outputType).name), output[i])
                    self.progressBar.setValue(self.progressBar.value() + int(i/stackSize*20))
                    
                print("\nFINISHED writing stack to '{}#.{}'".format(fileName, OutputType(outputType).name))
            else:
                fileName = self.saveFileDialog("2D image (*.{})".format( OutputType(outputType).name))
                fileName = fileName.replace("." + OutputType(outputType).name, "")
                if fileName == "" or fileName == None:
                    fileName = "output"
                    
                io.imsave(str("{}.{}").format(fileName, OutputType(outputType).name), output)
                print("\nFINISHED writing image to '{}.{}'".format(fileName, OutputType(outputType).name))
            
        #visualization            
        if(self.visualizeOutput.isChecked()):
            if(not isStack):
                self.static_canvasOutput2D.show()
                #self.mainLayout.setColumnMinimumWidth(2,490)
                self._static_axOutput.clear() 
                self._static_axOutput.imshow(output)
                self._static_axOutput.axis('off')
                self._static_axOutput.figure.canvas.draw()              
                
            else:
                #visualization                        
                self.mayavi_output3D.show()
                
                newRGBList = []
                l = self.colormap.tolist()
                if(self.grayscaleOutput.isChecked()):
                    for i in range(0, len(l)):
                        newRGBList.append([i, i/255, i/255, i/255])     
                else:
                    for i in range(0, len(l)):
                        newRGBList.append([i, l[i][0]/255, l[i][1]/255, l[i][2]/255])   
                   
                    
                self.mayavi_output3D.visualization = VisualizationInput()
                self.mayavi_output3D.visualization.maxVal = 255
                self.mayavi_output3D.visualization.RGBcolorm = newRGBList
                self.mayavi_output3D.visualization.fullStack3D = grayColormap
                self.mayavi_output3D.refresh()
                
        self.progressBar.setValue(100)
        
#        self.mayavi_output3D3D.refresh()
        self.repaint()
        
    def saveFileDialog(self, fileTypeText):    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"Save output as","",fileTypeText, options=options)
        if fileName:
            print(fileName)
            
        return fileName
        
    def showErrorDialog(self, message="An error occurred", title="Error"):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


if __name__ == '__main__':
    try:
        app
    except NameError:
        app = QApplication([])
    else:
        print("QApplication already defined")
    
    racc = RACC()
    racc.show()
    app.exec_()
