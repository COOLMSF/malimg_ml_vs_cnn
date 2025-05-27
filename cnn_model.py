import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
from PyQt5.QtCore import QThread, pyqtSignal
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import io

class TrainingCallback(Callback):
    """用于在训练过程中更新进度的回调函数"""
    def __init__(self, update_progress_fn, update_epoch_fn):
        super().__init__()
        self.update_progress = update_progress_fn
        self.update_epoch = update_epoch_fn
        
    def on_epoch_begin(self, epoch, logs=None):
        self.update_progress(f"开始训练第 {epoch+1} 轮...")
        
    def on_epoch_end(self, epoch, logs=None):
        self.update_epoch(epoch, logs)
        self.update_progress(f"第 {epoch+1} 轮完成 - 损失: {logs.get('loss'):.4f}, 准确率: {logs.get('accuracy'):.4f}, 验证损失: {logs.get('val_loss'):.4f}, 验证准确率: {logs.get('val_accuracy'):.4f}")
        
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # 每10个批次更新一次进度
            self.update_progress(f"批次 {batch} - 损失: {logs.get('loss'):.4f}, 准确率: {logs.get('accuracy'):.4f}")

class CNNTrainer(QThread):
    """CNN深度学习模型训练线程"""
    update_progress = pyqtSignal(str)
    update_epoch = pyqtSignal(int, dict)
    update_metrics = pyqtSignal(dict)
    update_plots = pyqtSignal(dict)
    training_finished = pyqtSignal(str)
    
    def __init__(self, params, X_train, X_test, y_train, y_test, class_names):
        super().__init__()
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.model = None
        self.history = None
        
    def run(self):
        """训练CNN模型并计算性能指标"""
        self.update_progress.emit("开始训练CNN深度学习模型...")
        
        # 获取参数
        architecture = self.params['architecture']
        batch_size = self.params['batch_size']
        epochs = self.params['epochs']
        learning_rate = self.params['learning_rate']
        optimizer_name = self.params['optimizer']
        use_augmentation = self.params['use_augmentation']
        
        # 创建模型
        self.update_progress.emit(f"创建{architecture}模型...")
        
        # 获取输入形状
        input_shape = self.X_train.shape[1:]
        num_classes = len(np.unique(self.y_train))
        
        # 将标签转换为one-hot编码
        y_train_cat = tf.keras.utils.to_categorical(self.y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(self.y_test, num_classes)
        
        # 根据选择的架构创建模型
        if architecture == "简单CNN":
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        
        elif architecture == "LeNet-5":
            self.model = Sequential([
                Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(16, (5, 5), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(120, activation='relu'),
                Dense(84, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
        
        elif architecture == "自定义CNN":
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Conv2D(64, (3, 3), activation='relu'),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        
        # 选择优化器
        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 模型摘要
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        self.update_progress.emit("\n".join(model_summary))
        
        # 数据增强
        if use_augmentation:
            self.update_progress.emit("使用数据增强...")
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )
            datagen.fit(self.X_train)
            
            # 创建训练回调
            callback = TrainingCallback(
                update_progress_fn=lambda x: self.update_progress.emit(x),
                update_epoch_fn=lambda epoch, logs: self.update_epoch.emit(epoch, logs)
            )
            
            # 训练模型
            start_time = time.time()
            self.update_progress.emit("开始训练模型...")
            
            self.history = self.model.fit(
                datagen.flow(self.X_train, y_train_cat, batch_size=batch_size),
                epochs=epochs,
                validation_data=(self.X_test, y_test_cat),
                callbacks=[callback]
            )
        else:
            # 创建训练回调
            callback = TrainingCallback(
                update_progress_fn=lambda x: self.update_progress.emit(x),
                update_epoch_fn=lambda epoch, logs: self.update_epoch.emit(epoch, logs)
            )
            
            # 训练模型
            start_time = time.time()
            self.update_progress.emit("开始训练模型...")
            
            self.history = self.model.fit(
                self.X_train, y_train_cat,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.X_test, y_test_cat),
                callbacks=[callback]
            )
        
        training_time = time.time() - start_time
        self.update_progress.emit(f"模型训练完成，耗时: {training_time:.2f}秒")
        
        # 评估模型
        self.update_progress.emit("评估模型性能...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, y_test_cat)
        
        # 获取预测结果
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_decoded = np.argmax(y_test_cat, axis=1)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test_decoded, y_pred)
        
        # 获取分类报告
        report = classification_report(y_test_decoded, y_pred, output_dict=True)
        
        # 提取性能指标
        metrics = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'precision': np.mean([report[str(i)]['precision'] for i in range(num_classes) if str(i) in report]),
            'recall': np.mean([report[str(i)]['recall'] for i in range(num_classes) if str(i) in report]),
            'f1': np.mean([report[str(i)]['f1-score'] for i in range(num_classes) if str(i) in report]),
            'training_time': training_time
        }
        
        self.update_metrics.emit(metrics)
        self.update_progress.emit(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}")
        
        # 保存模型
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"cnn_{architecture.lower().replace('-', '_').replace(' ', '_')}_{int(time.time())}.keras"
        model_path = os.path.join(model_dir, model_filename)
        
        # 保存Keras模型 (添加.keras扩展名以兼容Keras 3.x)
        self.model.save(model_path)
        
        self.update_progress.emit(f"模型已保存至: {model_path}")
        
        # 生成并保存可视化图表
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 训练历史图
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        history_filename = os.path.join(plots_dir, f"cnn_{architecture.lower().replace('-', '_').replace(' ', '_')}_history_{int(time.time())}.png")
        plt.savefig(history_filename)
        plt.close()
        
        # 混淆矩阵图
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        cm_filename = os.path.join(plots_dir, f"cnn_{architecture.lower().replace('-', '_').replace(' ', '_')}_cm_{int(time.time())}.png")
        plt.savefig(cm_filename)
        plt.close()
        
        # 发送可视化图表路径
        plots = {
            'training_history': history_filename,
            'confusion_matrix': cm_filename
        }
        
        self.update_plots.emit(plots)
        self.training_finished.emit(model_path)

def create_simple_cnn(input_shape, num_classes):
    """创建简单CNN模型"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_lenet5(input_shape, num_classes):
    """创建LeNet-5模型"""
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_custom_cnn(input_shape, num_classes):
    """创建自定义CNN模型"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
