import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import pickle
import time
from PyQt5.QtCore import QThread, pyqtSignal

class TraditionalMLTrainer(QThread):
    """传统机器学习模型训练线程"""
    update_progress = pyqtSignal(str)
    update_metrics = pyqtSignal(dict)
    update_plots = pyqtSignal(dict)
    training_finished = pyqtSignal(str)
    
    def __init__(self, model_type, params, X_train, X_test, y_train, y_test, class_names):
        super().__init__()
        self.model_type = model_type
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.model = None
        
    def run(self):
        """训练模型并计算性能指标"""
        self.update_progress.emit("开始训练传统机器学习模型...")
        
        # 创建模型
        if self.model_type == "支持向量机 (SVM)":
            self.update_progress.emit(f"创建SVM模型，核函数: {self.params['kernel']}, C值: {self.params['C']}")
            self.model = SVC(kernel=self.params['kernel'], C=self.params['C'], probability=True)
        
        elif self.model_type == "随机森林 (Random Forest)":
            self.update_progress.emit(f"创建随机森林模型，树数量: {self.params['n_estimators']}, 最大深度: {self.params['max_depth']}")
            self.model = RandomForestClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                random_state=42
            )
        
        elif self.model_type == "K近邻 (KNN)":
            self.update_progress.emit(f"创建KNN模型，邻居数量: {self.params['n_neighbors']}")
            self.model = KNeighborsClassifier(n_neighbors=self.params['n_neighbors'])
        
        elif self.model_type == "决策树 (Decision Tree)":
            self.update_progress.emit("创建决策树模型")
            self.model = DecisionTreeClassifier(random_state=42)
        
        elif self.model_type == "朴素贝叶斯 (Naive Bayes)":
            self.update_progress.emit("创建朴素贝叶斯模型")
            self.model = GaussianNB()
        
        # 训练模型
        start_time = time.time()
        self.update_progress.emit("训练模型中...")
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        self.update_progress.emit(f"模型训练完成，耗时: {training_time:.2f}秒")
        
        # 预测和评估
        self.update_progress.emit("评估模型性能...")
        y_pred = self.model.predict(self.X_test)
        
        # 计算性能指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time
        }
        
        self.update_metrics.emit(metrics)
        self.update_progress.emit(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        
        # 生成混淆矩阵
        self.update_progress.emit("生成混淆矩阵...")
        cm = confusion_matrix(self.y_test, y_pred)
        
        # 生成ROC曲线数据
        self.update_progress.emit("生成ROC曲线...")
        n_classes = len(np.unique(self.y_train))
        
        # 二值化标签
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_train))
        
        # 获取预测概率
        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(self.X_test)
        else:
            # 对于不支持predict_proba的模型，使用decision_function
            if hasattr(self.model, "decision_function"):
                y_score = self.model.decision_function(self.X_test)
                if y_score.ndim == 1:
                    y_score = np.column_stack([1 - y_score, y_score])
            else:
                # 如果两者都不支持，则使用伪概率
                y_score = np.zeros((len(self.y_test), n_classes))
                for i, pred in enumerate(y_pred):
                    y_score[i, pred] = 1
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            if i < y_test_bin.shape[1] and i < y_score.shape[1]:
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 保存模型
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{self.model_type.split(' ')[0].lower()}_{int(time.time())}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.update_progress.emit(f"模型已保存至: {model_path}")
        
        # 生成并保存可视化图表
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 混淆矩阵图
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')  # 使用英文标题避免中文字体问题
        plt.colorbar()
        plt.tight_layout()
        cm_filename = os.path.join(plots_dir, f"{self.model_type.split(' ')[0].lower()}_cm_{int(time.time())}.png")
        plt.savefig(cm_filename)
        plt.close()
        
        # ROC曲线图
        plt.figure(figsize=(10, 8))
        for i in range(min(5, n_classes)):  # 只显示前5个类别的ROC曲线，避免过于拥挤
            if i in fpr and i in tpr and i in roc_auc:
                plt.plot(fpr[i], tpr[i], lw=2,
                        label=f'Class {i} (AUC = {roc_auc[i]:.2f})')  # 使用英文标签避免中文字体问题
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')  # 使用英文标签避免中文字体问题
        plt.ylabel('True Positive Rate')   # 使用英文标签避免中文字体问题
        plt.title('ROC Curve')             # 使用英文标签避免中文字体问题
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_filename = os.path.join(plots_dir, f"{self.model_type.split(' ')[0].lower()}_roc_{int(time.time())}.png")
        plt.savefig(roc_filename)
        plt.close()
        
        # 发送可视化图表路径
        plots = {
            'confusion_matrix': cm_filename,
            'roc_curve': roc_filename
        }
        
        self.update_plots.emit(plots)
        self.training_finished.emit(model_path)
