import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton, QCheckBox,
                            QProgressBar, QTextEdit, QSplitter, QFrame, QGridLayout,
                            QScrollArea, QSlider, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QFont, QImage
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from traditional_ml import TraditionalMLTrainer
from cnn_model import CNNTrainer
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import label_binarize

class MalwareDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("恶意软件检测系统 - 传统机器学习与CNN深度学习对照实验")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载数据
        self.load_data()
        
        # 创建主窗口部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 初始化模型列表 - 修复：确保在使用前初始化这些属性
        self.ml_models = {}
        self.cnn_models = {}
        self.update_model_lists()
        
        # 创建各个标签页
        self.setup_dataset_tab()
        self.setup_traditional_ml_tab()
        self.setup_cnn_tab()
        self.setup_comparison_tab()
        self.setup_detection_tab()
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def load_data(self):
        """加载预处理后的数据"""
        try:
            data = np.load('processed_data.npz')
            self.X_train_cnn = data['X_train_cnn']
            self.X_test_cnn = data['X_test_cnn']
            self.X_train_ml = data['X_train_ml']
            self.X_test_ml = data['X_test_ml']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.class_names = data['class_names']
            
            print(f"数据加载成功: X_train_cnn shape: {self.X_train_cnn.shape}")
            print(f"X_test_cnn shape: {self.X_test_cnn.shape}")
            print(f"X_train_ml shape: {self.X_train_ml.shape}")
            print(f"X_test_ml shape: {self.X_test_ml.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"y_test shape: {self.y_test.shape}")
            print(f"类别数量: {len(self.class_names)}")
        except Exception as e:
            print(f"加载数据出错: {e}")
            # 创建空数据，以便应用程序仍然可以启动
            self.X_train_cnn = np.array([])
            self.X_test_cnn = np.array([])
            self.X_train_ml = np.array([])
            self.X_test_ml = np.array([])
            self.y_train = np.array([])
            self.y_test = np.array([])
            self.class_names = np.array([])
            
    def update_model_lists(self):
        """更新模型列表"""
        model_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 清空当前模型列表
        self.ml_models = {}
        self.cnn_models = {}
        
        # 扫描模型目录
        for filename in os.listdir(model_dir):
            filepath = os.path.join(model_dir, filename)
            if filename.endswith('.pkl'):
                # 传统机器学习模型
                model_name = filename.split('_')[0].capitalize()
                self.ml_models[filename] = filepath
            elif filename.endswith('.keras') and filename.startswith('cnn_'):
                # CNN模型 (Keras 3.x格式，.keras扩展名)
                model_name = filename.split('_')[1].capitalize()
                self.cnn_models[filename] = filepath
            elif os.path.isdir(filepath) and filename.startswith('cnn_'):
                # 兼容旧版CNN模型目录格式
                model_name = filename.split('_')[1].capitalize()
                self.cnn_models[filename] = filepath
                
        # 更新界面上的模型列表
        if hasattr(self, 'ml_model_list'):
            self.ml_model_list.clear()
            self.ml_model_list.addItems(list(self.ml_models.keys()))
            
        if hasattr(self, 'cnn_model_list'):
            self.cnn_model_list.clear()
            self.cnn_model_list.addItems(list(self.cnn_models.keys()))
            
        if hasattr(self, 'detection_model_type') and hasattr(self, 'detection_model_list'):
            self.detection_model_list.clear()
            if self.detection_model_type.currentText() == "传统机器学习模型":
                self.detection_model_list.addItems(list(self.ml_models.keys()))
            else:
                self.detection_model_list.addItems(list(self.cnn_models.keys()))
        
    def setup_dataset_tab(self):
        """设置数据集标签页"""
        dataset_tab = QWidget()
        layout = QVBoxLayout(dataset_tab)
        
        # 数据集信息区域
        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # 构建数据集信息文本
        dataset_info = "Malimg数据集信息:\n\n"
        dataset_info += f"总样本数: {len(self.y_train) + len(self.y_test)}\n"
        dataset_info += f"训练集样本数: {len(self.y_train)}\n"
        dataset_info += f"测试集样本数: {len(self.y_test)}\n"
        dataset_info += f"图像尺寸: {self.X_train_cnn.shape[1]}x{self.X_train_cnn.shape[2]}\n"
        dataset_info += f"类别数量: {len(self.class_names)}\n\n"
        
        # 添加类别分布信息
        dataset_info += "类别分布:\n"
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique_labels, counts):
            dataset_info += f"类别 {label}: {count} 个样本\n"
            
        info_text.setText(dataset_info)
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
        self.ml_model_combo.currentIndexChanged.connect(self.update_ml_params_visibility)
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
        self.ml_train_btn.clicked.connect(self.train_ml_model)
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
        
        # 初始化参数可见性
        self.update_ml_params_visibility()
        
    def update_ml_params_visibility(self):
        """根据选择的模型类型更新参数可见性"""
        model_type = self.ml_model_combo.currentText()
        
        # 隐藏所有参数
        self.svm_kernel_combo.setVisible(False)
        self.svm_c_spin.setVisible(False)
        self.rf_n_estimators_spin.setVisible(False)
        self.rf_max_depth_spin.setVisible(False)
        self.knn_n_neighbors_spin.setVisible(False)
        
        # 根据模型类型显示相应参数
        if "SVM" in model_type:
            self.svm_kernel_combo.setVisible(True)
            self.svm_c_spin.setVisible(True)
        elif "Random Forest" in model_type:
            self.rf_n_estimators_spin.setVisible(True)
            self.rf_max_depth_spin.setVisible(True)
        elif "KNN" in model_type:
            self.knn_n_neighbors_spin.setVisible(True)
            
    def train_ml_model(self):
        """训练传统机器学习模型"""
        if len(self.X_train_ml) == 0 or len(self.y_train) == 0:
            QMessageBox.warning(self, "错误", "数据集未加载或为空")
            return
            
        model_type = self.ml_model_combo.currentText()
        
        # 获取模型参数
        params = {}
        if "SVM" in model_type:
            params['kernel'] = self.svm_kernel_combo.currentText()
            params['C'] = self.svm_c_spin.value()
        elif "Random Forest" in model_type:
            params['n_estimators'] = self.rf_n_estimators_spin.value()
            params['max_depth'] = self.rf_max_depth_spin.value()
        elif "KNN" in model_type:
            params['n_neighbors'] = self.knn_n_neighbors_spin.value()
            
        # 禁用训练按钮
        self.ml_train_btn.setEnabled(False)
        self.ml_train_btn.setText("训练中...")
        
        # 清空性能指标
        self.ml_metrics_text.clear()
        self.ml_metrics_text.append("开始训练...")
        
        # 创建并启动训练线程
        self.ml_trainer = TraditionalMLTrainer(
            model_type=model_type,
            params=params,
            X_train=self.X_train_ml,
            X_test=self.X_test_ml,
            y_train=self.y_train,
            y_test=self.y_test,
            class_names=self.class_names
        )
        
        # 连接信号
        self.ml_trainer.update_progress.connect(self.update_ml_progress)
        self.ml_trainer.update_metrics.connect(self.update_ml_metrics)
        self.ml_trainer.update_plots.connect(self.update_ml_plots)
        self.ml_trainer.training_finished.connect(self.ml_training_finished)
        
        # 启动训练
        self.ml_trainer.start()
        
    def update_ml_progress(self, message):
        """更新训练进度信息"""
        self.ml_metrics_text.append(message)
        self.statusBar().showMessage(message)
        
    def update_ml_metrics(self, metrics):
        """更新性能指标"""
        metrics_text = "性能指标:\n"
        metrics_text += f"准确率: {metrics['accuracy']:.4f}\n"
        metrics_text += f"精确率: {metrics['precision']:.4f}\n"
        metrics_text += f"召回率: {metrics['recall']:.4f}\n"
        metrics_text += f"F1分数: {metrics['f1']:.4f}\n"
        metrics_text += f"训练时间: {metrics['training_time']:.2f}秒\n"
        
        self.ml_metrics_text.append(metrics_text)
        
    def update_ml_plots(self, plots):
        """更新可视化图表"""
        # 创建水平布局
        viz_layout = QHBoxLayout()
        
        # 混淆矩阵图
        cm_label = QLabel()
        cm_pixmap = QPixmap(plots['confusion_matrix'])
        cm_label.setPixmap(cm_pixmap.scaled(500, 400, Qt.KeepAspectRatio))
        cm_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(cm_label)
        
        # ROC曲线图
        roc_label = QLabel()
        roc_pixmap = QPixmap(plots['roc_curve'])
        roc_label.setPixmap(roc_pixmap.scaled(500, 400, Qt.KeepAspectRatio))
        roc_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(roc_label)
        
        # 修复：添加空值检查，确保parent()和layout()不为None
        parent = self.ml_viz_label.parent()
        if parent is None:
            # 如果parent为None，重新将标签添加到原始布局
            viz_group = QGroupBox("性能可视化")
            viz_layout_container = QVBoxLayout()
            viz_layout_container.addLayout(viz_layout)
            viz_group.setLayout(viz_layout_container)
            
            # 找到传统机器学习标签页
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "传统机器学习":
                    tab_widget = self.tabs.widget(i)
                    if tab_widget and tab_widget.layout():
                        # 添加到标签页布局的末尾
                        tab_widget.layout().addWidget(viz_group, 2)
                        break
            return
            
        # 如果parent存在，检查layout
        parent_layout = parent.layout()
        if parent_layout is None:
            # 如果layout为None，创建新布局
            parent_layout = QVBoxLayout()
            parent.setLayout(parent_layout)
        
        # 清除原有内容并设置新布局
        # 删除原有的标签
        self.ml_viz_label.setParent(None)
        
        # 清除原有布局中的所有部件
        while parent_layout.count():
            item = parent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # 设置新布局
        parent_layout.addLayout(viz_layout)
        
    def ml_training_finished(self, model_path):
        """训练完成后的处理"""
        self.ml_train_btn.setEnabled(True)
        self.ml_train_btn.setText("训练模型")
        self.statusBar().showMessage(f"训练完成，模型已保存至: {model_path}")
        
        # 更新模型列表
        self.update_model_lists()
        
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
        self.cnn_train_btn.clicked.connect(self.train_cnn_model)
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
        
    def train_cnn_model(self):
        """训练CNN深度学习模型"""
        if len(self.X_train_cnn) == 0 or len(self.y_train) == 0:
            QMessageBox.warning(self, "错误", "数据集未加载或为空")
            return
            
        # 获取模型参数
        params = {
            'architecture': self.cnn_arch_combo.currentText(),
            'batch_size': self.cnn_batch_size_spin.value(),
            'epochs': self.cnn_epochs_spin.value(),
            'learning_rate': self.cnn_lr_spin.value(),
            'optimizer': self.cnn_optimizer_combo.currentText(),
            'use_augmentation': self.cnn_augmentation_check.isChecked()
        }
        
        # 禁用训练按钮
        self.cnn_train_btn.setEnabled(False)
        self.cnn_train_btn.setText("训练中...")
        
        # 清空日志和进度条
        self.cnn_log_text.clear()
        self.cnn_log_text.append("开始训练...")
        self.cnn_progress_bar.setValue(0)
        
        # 创建并启动训练线程
        self.cnn_trainer = CNNTrainer(
            params=params,
            X_train=self.X_train_cnn,
            X_test=self.X_test_cnn,
            y_train=self.y_train,
            y_test=self.y_test,
            class_names=self.class_names
        )
        
        # 连接信号
        self.cnn_trainer.update_progress.connect(self.update_cnn_progress)
        self.cnn_trainer.update_epoch.connect(self.update_cnn_epoch)
        self.cnn_trainer.update_metrics.connect(self.update_cnn_metrics)
        self.cnn_trainer.update_plots.connect(self.update_cnn_plots)
        self.cnn_trainer.training_finished.connect(self.cnn_training_finished)
        
        # 启动训练
        self.cnn_trainer.start()
        
    def update_cnn_progress(self, message):
        """更新CNN训练进度信息"""
        self.cnn_log_text.append(message)
        self.statusBar().showMessage(message)
        
    def update_cnn_epoch(self, epoch, logs):
        """更新每个训练轮次的进度"""
        total_epochs = self.cnn_epochs_spin.value()
        progress = int((epoch + 1) / total_epochs * 100)
        self.cnn_progress_bar.setValue(progress)
        
    def update_cnn_metrics(self, metrics):
        """更新CNN性能指标"""
        metrics_text = "性能指标:\n"
        metrics_text += f"准确率: {metrics['accuracy']:.4f}\n"
        metrics_text += f"损失: {metrics['loss']:.4f}\n"
        metrics_text += f"精确率: {metrics['precision']:.4f}\n"
        metrics_text += f"召回率: {metrics['recall']:.4f}\n"
        metrics_text += f"F1分数: {metrics['f1']:.4f}\n"
        metrics_text += f"训练时间: {metrics['training_time']:.2f}秒\n"
        
        self.cnn_log_text.append(metrics_text)
        
    def update_cnn_plots(self, plots):
        """更新可视化图表"""
        # 创建水平布局
        viz_layout = QHBoxLayout()
        
        # 混淆矩阵图
        cm_label = QLabel()
        cm_pixmap = QPixmap(plots['confusion_matrix'])
        cm_label.setPixmap(cm_pixmap.scaled(500, 400, Qt.KeepAspectRatio))
        cm_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(cm_label)
        
        # 训练历史图
        history_label = QLabel()
        history_pixmap = QPixmap(plots['training_history'])
        history_label.setPixmap(history_pixmap.scaled(500, 400, Qt.KeepAspectRatio))
        history_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(history_label)
        
        # 修复：添加空值检查，确保parent()和layout()不为None
        parent = self.cnn_viz_label.parent()
        if parent is None:
            # 如果parent为None，重新将标签添加到原始布局
            viz_group = QGroupBox("性能可视化")
            viz_layout_container = QVBoxLayout()
            viz_layout_container.addLayout(viz_layout)
            viz_group.setLayout(viz_layout_container)
            
            # 找到CNN深度学习标签页
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "CNN深度学习":
                    tab_widget = self.tabs.widget(i)
                    if tab_widget and tab_widget.layout():
                        # 添加到标签页布局的末尾
                        tab_widget.layout().addWidget(viz_group, 2)
                        break
            return
            
        # 如果parent存在，检查layout
        parent_layout = parent.layout()
        if parent_layout is None:
            # 如果layout为None，创建新布局
            parent_layout = QVBoxLayout()
            parent.setLayout(parent_layout)
        
        # 清除原有内容并设置新布局
        # 删除原有的标签
        self.cnn_viz_label.setParent(None)
        
        # 清除原有布局中的所有部件
        while parent_layout.count():
            item = parent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # 设置新布局
        parent_layout.addLayout(viz_layout)        
    def cnn_training_finished(self, model_path):
        """CNN训练完成后的处理"""
        self.cnn_train_btn.setEnabled(True)
        self.cnn_train_btn.setText("训练模型")
        self.cnn_progress_bar.setValue(100)
        self.statusBar().showMessage(f"训练完成，模型已保存至: {model_path}")
        
        # 更新模型列表
        self.update_model_lists()
        
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
        compare_btn.clicked.connect(self.compare_models)
        
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
        
    def compare_models(self):
        """对比传统机器学习和CNN深度学习模型"""
        if not self.ml_model_list.count() or not self.cnn_model_list.count():
            QMessageBox.warning(self, "错误", "请先训练模型")
            return
            
        ml_model_name = self.ml_model_list.currentText()
        cnn_model_name = self.cnn_model_list.currentText()
        
        if not ml_model_name or not cnn_model_name:
            QMessageBox.warning(self, "错误", "请选择要对比的模型")
            return
            
        # 加载模型
        try:
            # 加载传统机器学习模型
            with open(self.ml_models[ml_model_name], 'rb') as f:
                ml_model = pickle.load(f)
                
            # 加载CNN模型
            cnn_model = load_model(self.cnn_models[cnn_model_name])
            
            # 评估传统机器学习模型
            ml_start_time = time.time()
            ml_y_pred = ml_model.predict(self.X_test_ml)
            ml_inference_time = time.time() - ml_start_time
            ml_accuracy = np.mean(ml_y_pred == self.y_test)
            
            # 评估CNN模型
            cnn_start_time = time.time()
            try:
                cnn_y_pred_prob = cnn_model.predict(self.X_test_cnn)
                cnn_inference_time = time.time() - cnn_start_time
                cnn_y_pred = np.argmax(cnn_y_pred_prob, axis=1)
                cnn_accuracy = np.mean(cnn_y_pred == self.y_test)
            except Exception as e:
                self.comparison_text.setText(f"CNN模型评估出错: {str(e)}")
                return
            
            # 显示对比结果
            comparison_text = "模型对比结果:\n\n"
            comparison_text += f"传统机器学习模型: {ml_model_name}\n"
            comparison_text += f"CNN深度学习模型: {cnn_model_name}\n\n"
            comparison_text += f"传统机器学习模型准确率: {ml_accuracy:.4f}\n"
            comparison_text += f"CNN深度学习模型准确率: {cnn_accuracy:.4f}\n\n"
            comparison_text += f"传统机器学习模型推理时间: {ml_inference_time:.4f}秒\n"
            comparison_text += f"CNN深度学习模型推理时间: {cnn_inference_time:.4f}秒\n"
            
            self.comparison_text.setText(comparison_text)
            
            # 创建对比可视化
            self.create_comparison_visualization(ml_accuracy, cnn_accuracy, ml_inference_time, cnn_inference_time)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"对比模型时出错: {str(e)}")
            
    def create_comparison_visualization(self, ml_accuracy, cnn_accuracy, ml_inference_time, cnn_inference_time):
        """创建模型对比可视化"""
        # 创建图表
        plt.figure(figsize=(12, 5))
        
        # 准确率对比
        plt.subplot(1, 2, 1)
        models = ['传统机器学习', 'CNN深度学习']
        accuracies = [ml_accuracy, cnn_accuracy]
        plt.bar(models, accuracies, color=['blue', 'orange'])
        plt.title('准确率对比')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        
        # 推理时间对比
        plt.subplot(1, 2, 2)
        times = [ml_inference_time, cnn_inference_time]
        plt.bar(models, times, color=['blue', 'orange'])
        plt.title('推理时间对比')
        plt.ylabel('时间(秒)')
        
        plt.tight_layout()
        
        # 保存图表
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        comparison_filename = os.path.join(plots_dir, f"comparison_{int(time.time())}.png")
        plt.savefig(comparison_filename)
        plt.close()
        
        # 显示图表
        comparison_label = QLabel()
        comparison_pixmap = QPixmap(comparison_filename)
        comparison_label.setPixmap(comparison_pixmap.scaled(800, 400, Qt.KeepAspectRatio))
        comparison_label.setAlignment(Qt.AlignCenter)
        
        # 修复：添加空值检查，确保parent()和layout()不为None
        parent = self.comparison_viz_label.parent()
        if parent is None:
            # 如果parent为None，重新将标签添加到原始布局
            viz_group = QGroupBox("性能对比可视化")
            viz_layout = QVBoxLayout()
            viz_layout.addWidget(comparison_label)
            viz_group.setLayout(viz_layout)
            
            # 找到对比分析标签页
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "对比分析":
                    tab_widget = self.tabs.widget(i)
                    if tab_widget and tab_widget.layout():
                        # 添加到标签页布局的末尾
                        tab_widget.layout().addWidget(viz_group, 2)
                        break
            return
            
        # 如果parent存在，检查layout
        parent_layout = parent.layout()
        if parent_layout is None:
            # 如果layout为None，创建新布局
            parent_layout = QVBoxLayout()
            parent.setLayout(parent_layout)
            parent_layout.addWidget(comparison_label)
            return
        
        # 清除原有内容并设置新布局
        # 删除原有的标签
        self.comparison_viz_label.setParent(None)
        
        # 清除原有布局中的所有部件
        while parent_layout.count():
            item = parent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # 设置新布局
        parent_layout.addWidget(comparison_label)
        
    def setup_detection_tab(self):
        """设置恶意软件检测标签页"""
        detection_tab = QWidget()
        layout = QVBoxLayout(detection_tab)
        
        # 模型选择区域
        model_group = QGroupBox("检测模型选择")
        model_layout = QHBoxLayout()
        
        self.detection_model_type = QComboBox()
        self.detection_model_type.addItems(["传统机器学习模型", "CNN深度学习模型"])
        self.detection_model_type.currentIndexChanged.connect(self.update_detection_model_list)
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
        upload_btn.clicked.connect(self.select_malware_file)
        upload_layout.addWidget(upload_btn)
        
        detect_btn = QPushButton("开始检测")
        detect_btn.clicked.connect(self.detect_malware)
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
        
        # 初始化模型列表
        self.update_detection_model_list()
        
    def update_detection_model_list(self):
        """根据选择的模型类型更新检测模型列表"""
        self.detection_model_list.clear()
        
        if self.detection_model_type.currentText() == "传统机器学习模型":
            self.detection_model_list.addItems(list(self.ml_models.keys()))
        else:
            self.detection_model_list.addItems(list(self.cnn_models.keys()))
            
    def select_malware_file(self):
        """选择恶意软件文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "所有文件 (*)")
        
        if file_path:
            self.file_path_label.setText(file_path)
            
    def detect_malware(self):
        """检测恶意软件"""
        file_path = self.file_path_label.text()
        
        if file_path == "未选择文件":
            QMessageBox.warning(self, "错误", "请先选择文件")
            return
            
        if not self.detection_model_list.count():
            QMessageBox.warning(self, "错误", "请先训练模型")
            return
            
        model_type = self.detection_model_type.currentText()
        model_name = self.detection_model_list.currentText()
        
        if not model_name:
            QMessageBox.warning(self, "错误", "请选择检测模型")
            return
            
        # 清空结果
        self.detection_result_text.clear()
        self.detection_result_text.append("开始检测...")
        
        try:
            # 读取文件并转换为图像
            with open(file_path, 'rb') as f:
                file_data = f.read()
                
            # 将文件数据转换为32x32灰度图像
            # 这里简化处理，实际应用中需要根据具体情况进行适当的预处理
            image_data = np.frombuffer(file_data[:1024], dtype=np.uint8)
            if len(image_data) < 1024:
                # 如果数据不足，填充0
                image_data = np.pad(image_data, (0, 1024 - len(image_data)), 'constant')
            
            # 重塑为32x32
            image = image_data.reshape(32, 32)
            
            # 归一化
            image = image.astype('float32') / 255.0
            
            # 加载模型并预测
            if model_type == "传统机器学习模型":
                # 重塑为一维数组
                image_flat = image.reshape(1, -1)
                
                # 加载传统机器学习模型
                with open(self.ml_models[model_name], 'rb') as f:
                    model = pickle.load(f)
                    
                # 预测
                prediction = model.predict(image_flat)[0]
                
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(image_flat)[0]
                    confidence = probabilities[prediction]
                else:
                    confidence = None
                    
            else:
                # 重塑为CNN输入格式
                image_cnn = image.reshape(1, 32, 32, 1)
                
                # 加载CNN模型
                try:
                    # 尝试直接加载模型文件
                    model = load_model(self.cnn_models[model_name])
                except:
                    # 如果失败，可能是旧格式目录，尝试添加扩展名
                    if not self.cnn_models[model_name].endswith('.keras'):
                        model_path = self.cnn_models[model_name] + '.keras'
                        if os.path.exists(model_path):
                            model = load_model(model_path)
                        else:
                            raise ValueError(f"无法加载模型: {self.cnn_models[model_name]}")
                
                # 预测
                pred_probs = model.predict(image_cnn)[0]
                prediction = np.argmax(pred_probs)
                confidence = pred_probs[prediction]
                
            # 显示结果
            result_text = f"检测结果:\n\n"
            result_text += f"文件: {os.path.basename(file_path)}\n"
            result_text += f"模型: {model_name}\n"
            result_text += f"预测类别: {prediction}\n"
            
            if confidence is not None:
                result_text += f"置信度: {confidence:.4f}\n"
                
            self.detection_result_text.setText(result_text)
            
        except Exception as e:
            self.detection_result_text.setText(f"检测出错: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MalwareDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
