import numpy as np

# 加载数据集
data = np.load('malimg.npz', allow_pickle=True)
arr = data['arr']

# 分析数据结构
print('Type of arr:', type(arr))
print('Shape of arr:', arr.shape if hasattr(arr, 'shape') else 'No shape attribute')
print('Length of arr:', len(arr))
print('First element type:', type(arr[0]) if len(arr) > 0 else 'Empty array')

# 查看前几个元素
print('\nSample of first few elements:')
for i in range(min(3, len(arr))):
    print(f'Element {i} type:', type(arr[i]))
    print(f'Element {i} shape (if applicable):', arr[i].shape if hasattr(arr[i], 'shape') else 'No shape attribute')
    print(f'Element {i} content preview:', arr[i])
    print('-' * 50)

# 如果arr是结构化数组，尝试查看其字段
if hasattr(arr.dtype, 'names') and arr.dtype.names is not None:
    print('\nStructured array fields:', arr.dtype.names)
    for field in arr.dtype.names:
        print(f'Field {field} sample:', arr[field][0])
