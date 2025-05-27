import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# 创建保存图像的目录
os.makedirs('sample_images', exist_ok=True)

# 加载数据集
print("加载数据集...")
data = np.load('malimg.npz', allow_pickle=True)
arr = data['arr']

# 提取特征和标签
print("提取特征和标签...")
X = np.array([sample[0] for sample in arr])
y = np.array([sample[1] for sample in arr])

# 检查数据形状
print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"标签唯一值: {np.unique(y)}")

# 统计每个类别的样本数量
unique_labels, counts = np.unique(y, return_counts=True)
print("各类别样本数量:")
for label, count in zip(unique_labels, counts):
    print(f"类别 {label}: {count} 个样本")

# 可视化一些样本图像
print("可视化样本图像...")
plt.figure(figsize=(15, 10))
for i in range(min(10, len(X))):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f"类别: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images/sample_malware_images.png')
plt.close()

# 数据预处理
print("数据预处理...")
# 归一化图像数据
X_normalized = X.astype('float32') / 255.0

# 重塑数据以适应CNN (样本数, 高度, 宽度, 通道数)
X_reshaped_cnn = X_normalized.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# 重塑数据以适应传统机器学习 (样本数, 特征数)
X_reshaped_ml = X_normalized.reshape(X.shape[0], -1)

# 划分训练集和测试集
print("划分训练集和测试集...")
X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(
    X_reshaped_cnn, y, test_size=0.2, random_state=42, stratify=y
)
X_train_ml, X_test_ml = X_train_cnn.reshape(X_train_cnn.shape[0], -1), X_test_cnn.reshape(X_test_cnn.shape[0], -1)

# 保存处理后的数据
print("保存处理后的数据...")
np.savez_compressed('processed_data.npz', 
                    X_train_cnn=X_train_cnn, 
                    X_test_cnn=X_test_cnn, 
                    X_train_ml=X_train_ml, 
                    X_test_ml=X_test_ml, 
                    y_train=y_train, 
                    y_test=y_test,
                    class_names=unique_labels)

print("数据预处理完成!")
