import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton, QCheckBox,
                            QProgressBar, QTextEdit, QSplitter, QFrame, QGridLayout,
                            QScrollArea, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

class MalwareDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("恶意软件检测系统 - 传统机器学习与CNN深度学习对照实验")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 创建各个标签页
        self.setup_dataset_tab()
        self.setup_traditional_ml_tab()
        self.setup_cnn_tab()
        self.setup_comparison_tab()
        self.setup_detection_tab()
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def setup_dataset_tab(self):
        """设置数据集标签页"""
        dataset_tab = QWidget()
        layout = QVBoxLayout(dataset_tab)
        
        # 数据集信息区域
        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setText("Malimg数据集包含9339个样本，每个样本是32x32像素的灰度图像，共有25个不同的恶意软件类别。")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        
        # 数据可视化区域
        viz_group = QGroupBox("数据可视化")
        viz_layout = QVBoxLayout()
        
        # 添加样本图像
        sample_label = QLabel()
        sample_path = os.path.join(os.getcwd(), "sample_images", "sample_malware_images.png")
        if os.path.exists(sample_path):
            pixmap = QPixmap(sample_path)
            sample_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
            sample_label.setAlignment(Qt.AlignCenter)
        else:
            sample_label.setText("样本图像未找到")
        
        viz_layout.addWidget(sample_label)
        viz_group.setLayout(viz_layout)
        
        # 添加到布局
        layout.addWidget(info_group, 1)
        layout.addWidget(viz_group, 3)
        
        self.tabs.addTab(dataset_tab, "数据集")
        
    def setup_traditional_ml_tab(self):
        """设置传统机器学习标签页"""
        ml_tab = QWidget()
        layout = QVBoxLayout(ml_tab)
        
        # 参数选择区域
        param_group = QGroupBox("模型参数")
        param_layout = QGridLayout()
        
        # 模型选择
        param_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.ml_model_combo = QComboBox()
        self.ml_model_combo.addItems(["支持向量机 (SVM)", "随机森林 (Random Forest)", 
                                     "K近邻 (KNN)", "决策树 (Decision Tree)", 
                                     "朴素贝叶斯 (Naive Bayes)"])
        param_layout.addWidget(self.ml_model_combo, 0, 1)
        
        # SVM参数
        param_layout.addWidget(QLabel("核函数:"), 1, 0)
        self.svm_kernel_combo = QComboBox()
        self.svm_kernel_combo.addItems(["linear", "poly", "rbf", "sigmoid"])
        param_layout.addWidget(self.svm_kernel_combo, 1, 1)
        
        param_layout.addWidget(QLabel("C值:"), 2, 0)
        self.svm_c_spin = QDoubleSpinBox()
        self.svm_c_spin.setRange(0.1, 100.0)
        self.svm_c_spin.setValue(1.0)
        self.svm_c_spin.setSingleStep(0.1)
        param_layout.addWidget(self.svm_c_spin, 2, 1)
        
        # 随机森林参数
        param_layout.addWidget(QLabel("树的数量:"), 3, 0)
        self.rf_n_estimators_spin = QSpinBox()
        self.rf_n_estimators_spin.setRange(10, 500)
        self.rf_n_estimators_spin.setValue(100)
        self.rf_n_estimators_spin.setSingleStep(10)
        param_layout.addWidget(self.rf_n_estimators_spin, 3, 1)
        
        param_layout.addWidget(QLabel("最大深度:"), 4, 0)
        self.rf_max_depth_spin = QSpinBox()
        self.rf_max_depth_spin.setRange(1, 50)
        self.rf_max_depth_spin.setValue(10)
        param_layout.addWidget(self.rf_max_depth_spin, 4, 1)
        
        # KNN参数
        param_layout.addWidget(QLabel("邻居数量:"), 5, 0)
        self.knn_n_neighbors_spin = QSpinBox()
        self.knn_n_neighbors_spin.setRange(1, 20)
        self.knn_n_neighbors_spin.setValue(5)
        param_layout.addWidget(self.knn_n_neighbors_spin, 5, 1)
        
        # 训练按钮
        self.ml_train_btn = QPushButton("训练模型")
        param_layout.addWidget(self.ml_train_btn, 6, 0, 1, 2)
        
        param_group.setLayout(param_layout)
        
        # 性能指标区域
        metrics_group = QGroupBox("性能指标")
        metrics_layout = QVBoxLayout()
        
        self.ml_metrics_text = QTextEdit()
        self.ml_metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.ml_metrics_text)
        
        metrics_group.setLayout(metrics_layout)
        
        # 可视化区域
        viz_group = QGroupBox("性能可视化")
        viz_layout = QVBoxLayout()
        
        self.ml_viz_label = QLabel("训练后将显示混淆矩阵和ROC曲线")
        self.ml_viz_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.ml_viz_label)
        
        viz_group.setLayout(viz_layout)
        
        # 添加到布局
        h_layout = QHBoxLayout()
        h_layout.addWidget(param_group, 1)
        h_layout.addWidget(metrics_group, 2)
        
        layout.addLayout(h_layout, 1)
        layout.addWidget(viz_group, 2)
        
        self.tabs.addTab(ml_tab, "传统机器学习")
        
    def setup_cnn_tab(self):
        """设置CNN深度学习标签页"""
        cnn_tab = QWidget()
        layout = QVBoxLayout(cnn_tab)
        
        # 参数选择区域
        param_group = QGroupBox("模型参数")
        param_layout = QGridLayout()
        
        # CNN架构选择
        param_layout.addWidget(QLabel("CNN架构:"), 0, 0)
        self.cnn_arch_combo = QComboBox()
        self.cnn_arch_combo.addItems(["简单CNN", "LeNet-5", "自定义CNN"])
        param_layout.addWidget(self.cnn_arch_combo, 0, 1)
        
        # 训练参数
        param_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.cnn_batch_size_spin = QSpinBox()
        self.cnn_batch_size_spin.setRange(8, 256)
        self.cnn_batch_size_spin.setValue(32)
        self.cnn_batch_size_spin.setSingleStep(8)
        param_layout.addWidget(self.cnn_batch_size_spin, 1, 1)
        
        param_layout.addWidget(QLabel("训练轮数:"), 2, 0)
        self.cnn_epochs_spin = QSpinBox()
        self.cnn_epochs_spin.setRange(1, 100)
        self.cnn_epochs_spin.setValue(10)
        param_layout.addWidget(self.cnn_epochs_spin, 2, 1)
        
        param_layout.addWidget(QLabel("学习率:"), 3, 0)
        self.cnn_lr_spin = QDoubleSpinBox()
        self.cnn_lr_spin.setRange(0.0001, 0.1)
        self.cnn_lr_spin.setValue(0.001)
        self.cnn_lr_spin.setDecimals(4)
        self.cnn_lr_spin.setSingleStep(0.0001)
        param_layout.addWidget(self.cnn_lr_spin, 3, 1)
        
        # 优化器选择
        param_layout.addWidget(QLabel("优化器:"), 4, 0)
        self.cnn_optimizer_combo = QComboBox()
        self.cnn_optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        param_layout.addWidget(self.cnn_optimizer_combo, 4, 1)
        
        # 数据增强
        param_layout.addWidget(QLabel("数据增强:"), 5, 0)
        self.cnn_augmentation_check = QCheckBox("启用")
        param_layout.addWidget(self.cnn_augmentation_check, 5, 1)
        
        # 训练按钮
        self.cnn_train_btn = QPushButton("训练模型")
        param_layout.addWidget(self.cnn_train_btn, 6, 0, 1, 2)
        
        param_group.setLayout(param_layout)
        
        # 训练进度区域
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()
        
        self.cnn_progress_bar = QProgressBar()
        progress_layout.addWidget(self.cnn_progress_bar)
        
        self.cnn_log_text = QTextEdit()
        self.cnn_log_text.setReadOnly(True)
        progress_layout.addWidget(self.cnn_log_text)
        
        progress_group.setLayout(progress_layout)
        
        # 性能可视化区域
        viz_group = QGroupBox("训练过程与性能可视化")
        viz_layout = QVBoxLayout()
        
        self.cnn_viz_label = QLabel("训练后将显示损失曲线、准确率曲线和混淆矩阵")
        self.cnn_viz_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.cnn_viz_label)
        
        viz_group.setLayout(viz_layout)
        
        # 添加到布局
        h_layout = QHBoxLayout()
        h_layout.addWidget(param_group, 1)
        h_layout.addWidget(progress_group, 2)
        
        layout.addLayout(h_layout, 1)
        layout.addWidget(viz_group, 2)
        
        self.tabs.addTab(cnn_tab, "CNN深度学习")
        
    def setup_comparison_tab(self):
        """设置对比分析标签页"""
        comparison_tab = QWidget()
        layout = QVBoxLayout(comparison_tab)
        
        # 模型选择区域
        selection_group = QGroupBox("模型选择")
        selection_layout = QHBoxLayout()
        
        # 传统机器学习模型选择
        ml_group = QGroupBox("传统机器学习模型")
        ml_layout = QVBoxLayout()
        self.ml_model_list = QComboBox()
        ml_layout.addWidget(self.ml_model_list)
        ml_group.setLayout(ml_layout)
        
        # CNN模型选择
        cnn_group = QGroupBox("CNN深度学习模型")
        cnn_layout = QVBoxLayout()
        self.cnn_model_list = QComboBox()
        cnn_layout.addWidget(self.cnn_model_list)
        cnn_group.setLayout(cnn_layout)
        
        selection_layout.addWidget(ml_group)
        selection_layout.addWidget(cnn_group)
        selection_group.setLayout(selection_layout)
        
        # 对比按钮
        compare_btn = QPushButton("进行对比分析")
        
        # 对比结果区域
        result_group = QGroupBox("对比结果")
        result_layout = QVBoxLayout()
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        result_layout.addWidget(self.comparison_text)
        
        result_group.setLayout(result_layout)
        
        # 可视化区域
        viz_group = QGroupBox("性能对比可视化")
        viz_layout = QVBoxLayout()
        
        self.comparison_viz_label = QLabel("对比后将显示性能指标对比图")
        self.comparison_viz_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.comparison_viz_label)
        
        viz_group.setLayout(viz_layout)
        
        # 添加到布局
        layout.addWidget(selection_group)
        layout.addWidget(compare_btn)
        layout.addWidget(result_group, 1)
        layout.addWidget(viz_group, 2)
        
        self.tabs.addTab(comparison_tab, "对比分析")
        
    def setup_detection_tab(self):
        """设置恶意软件检测标签页"""
        detection_tab = QWidget()
        layout = QVBoxLayout(detection_tab)
        
        # 模型选择区域
        model_group = QGroupBox("检测模型选择")
        model_layout = QHBoxLayout()
        
        self.detection_model_type = QComboBox()
        self.detection_model_type.addItems(["传统机器学习模型", "CNN深度学习模型"])
        model_layout.addWidget(QLabel("模型类型:"))
        model_layout.addWidget(self.detection_model_type)
        
        self.detection_model_list = QComboBox()
        model_layout.addWidget(QLabel("具体模型:"))
        model_layout.addWidget(self.detection_model_list)
        
        model_group.setLayout(model_layout)
        
        # 文件上传区域
        upload_group = QGroupBox("恶意软件上传")
        upload_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("未选择文件")
        upload_layout.addWidget(self.file_path_label)
        
        upload_btn = QPushButton("选择文件")
        upload_layout.addWidget(upload_btn)
        
        detect_btn = QPushButton("开始检测")
        upload_layout.addWidget(detect_btn)
        
        upload_group.setLayout(upload_layout)
        
        # 检测结果区域
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        
        self.detection_result_text = QTextEdit()
        self.detection_result_text.setReadOnly(True)
        result_layout.addWidget(self.detection_result_text)
        
        result_group.setLayout(result_layout)
        
        # 添加到布局
        layout.addWidget(model_group)
        layout.addWidget(upload_group)
        layout.addWidget(result_group)
        
        self.tabs.addTab(detection_tab, "恶意软件检测")

def main():
    app = QApplication(sys.argv)
    window = MalwareDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
