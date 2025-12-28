# <center>YOLO：从零开始之部署准备</center>

**适用对象**：机器人竞赛视觉组、深度学习初学者
**核心目标**：搭建开发环境，完成你的第一个自定义数据集的训练。
**工具链**：Miniconda + PyTorch + YOLOv8 + X-AnyLabeling/Roboflow

---

## 🛠️ 第一部分：环境搭建 (基础建设)

### 1. 安装 Miniconda

不要安装几 GB 大小的 Anaconda，Miniconda 足够轻量。

* **下载**：推荐使用 [清华大学开源镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 下载最新版。
* **注意**：安装时**务必勾选** "Add Miniconda3 to my PATH environment variable"（即使安装程序提示不推荐，也要勾选，方便后续操作）。

### 2. 配置国内加速源

为了避免下载速度慢或连接超时，必须配置国内镜像。
在用户目录下找到 `.condarc` 文件（如果没有就新建一个 txt 改名），填入以下内容（当然最好科学上网）：

```yaml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
show_channel_urls: true

```

### 3. 创建虚拟环境与安装 PyTorch

打开 CMD 或 PowerShell，依次执行：

```bash
# 1. 初始化终端 (如果是 PowerShell 必须执行此步并重启终端，cmd不需要)
conda init cmd.exe
conda init powershell

# 2. 创建名为 yolo 的环境，指定 Python 3.13 (这里以3.13为例，用你自己的版本)
conda create -n yolo python=3.13 -y

# 3. 激活环境 (每次写代码前都要输这句)
conda activate yolo

# 4. 安装 PyTorch (根据硬件二选一)
# 选项 A: 有 NVIDIA 显卡
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 选项 B: 只有 Intel/AMD 核显 
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 4. 安装 YOLOv8 及修复 OpenCV

```bash
# 安装 Ultralytics (YOLOv8 核心库)
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可能存在的 OpenCV 无法弹窗的问题
# 先试运行一次，如果没有问题可忽略下面的步骤
# 先卸载可能自动安装的无头版
pip uninstall opencv-python opencv-python-headless -y
# 再安装完整版
pip install opencv-python

```

---

## 🏷️ 第二部分：数据标注 (X-AnyLabeling 工作流)

推荐使用 **X-AnyLabeling**，其具有 AI 辅助标注功能

### 1. 准备工作

* 新建项目文件夹（如 `E:\yolo_study`）。
* 新建 `dataset` 文件夹，放入采集好的图片（建议 100+ 张，覆盖不同光照、角度，当然初次配置一张就行）。
* 新建 `classes.txt`，第一行写入你的类别名（例如 `mouse` 或 `armor`）。

### 2. 标注流程

1. **导入**：打开软件 -> Open Dir (选 dataset) -> 导入 `classes.txt` (Import Class Names)。
2. **AI 预标注 (杀手级功能)**：
* 点击左侧“大脑”图标 -> Model -> Detection -> **YOLOv8n**。
* 按快捷键 `Ctrl + J`，AI 会自动帮你把图里的物体框出来。


3. **人工修正**：
* 检查 AI 标的框，错的删掉，漏的按 `R` 键手动补框。


4. **导出**：
* 点击顶部导出图标 (Export) -> 选择 **导出YOLO-Hbb标注**。
* 保存到新文件夹 `export_data`。



### 3. 数据集整理 (关键)

确保导出后的 `export_data` 文件夹结构如下（如果没有，请手动创建文件夹并移动文件）：

```text
export_data/
├── images/      # 所有的 .jpg 图片
├── labels/      # 所有的 .txt 标签
└── classes.txt

```

---

## 🏋️‍♂️ 第三部分：模型训练

### 1. 编写配置文件 (task.yaml)

在项目**根目录**新建 `mouse_task.yaml`。**建议使用绝对路径**以避免 `FileNotFoundError`（如果是协同性项目，则不建议使用绝对路径）。

```yaml
# 注意：Windows路径的反斜杠 \ 要改为正斜杠 /
path: E:/yolo_study/export_data  
train: images
val: images

nc: 1              # 类别数量
names: ['mouse']   # 类别名称

```

### 2. 编写训练脚本 (train.py)

```python
from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    model.train(
        data=r"E:\yolo_study\mouse_task.yaml", # 使用 r"" 包裹绝对路径
        epochs=20,     # 训练轮数 (建议 100-300，太少学不会，这里只是初始化，所以少)
        imgsz=640,     # 图片大小
        batch=16,      # 如果显存报错，改小这个数字
        workers=0      # Windows下必须设为0，否则报错
    )

```

### 3. 执行训练

在终端运行：`python train.py`。
等待训练完成，最好的模型会保存在 `runs/detect/train/weights/best.pt`。

---

## 👁️ 第四部分：验证与测试

使用 Python 脚本调用摄像头，验证模型效果。

### 编写推理脚本 (predict.py)

```python
import cv2
from ultralytics import YOLO

# ⚠️ 替换为你训练好的 best.pt 路径
model_path = r"E:\yolo_study\runs\detect\train\weights\best.pt"
conf_threshold = 0.5  # 置信度 (如果识别不到，改成 0.2 试试)

# 加载模型
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 打开摄像头 (0 代表默认摄像头)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 推理
    results = model.predict(frame, conf=conf_threshold, verbose=False)

    # 绘制结果
    annotated_frame = results[0].plot()

    # 显示画面
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

---

## 🚢 第五部分：为 C++ 部署做准备

为了在 Jetson Nano 或树莓派上高效运行，必须将模型转换为 ONNX 通用格式。

### 1. 安装导出工具

```bash
pip install onnx onnxsim

```

### 2. 一键导出

在终端运行：

```bash
# model 指向你训练好的 best.pt
yolo export model=runs/detect/train/weights/best.pt format=onnx

```

运行后，你将得到一个 **`best.onnx`** 文件。这就是通往 C++ 部署的“入场券”。

---

## ⚠️ 常见报错速查表 (FAQ)

1. **报错：`CondaError: Run 'conda init' before ...**`
* **原因**：PowerShell 权限或初始化问题。
* **解法**：运行 `conda init powershell`，然后**必须关闭并重新打开终端**。


2. **报错：`FileNotFoundError: ... .yaml does not exist**`
* **原因**：相对路径找不到文件。
* **解法**：在代码中始终使用**绝对路径**（例如 `E:/proj/data.yaml`）。


3. **报错：`cv2.error: The function is not implemented**`
* **原因**：安装了无头版 OpenCV (opencv-python-headless)。
* **解法**：`pip uninstall opencv-python-headless -y` 然后 `pip install opencv-python`。


4. **现象：训练完摄像头里完全没有框**
* **原因**：模型欠拟合。
* **解法**：
1. 检查数据量是否太少（<50张）。
2. 增加训练轮数（epochs 改为 100 或 300）。
3. 在推理代码中降低置信度阈值 (`conf=0.1`) 看看是不是因为模型不够自信。