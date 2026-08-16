<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD007 -->
<!-- markdownlint-disable MD005 -->
<!-- markdownlint-disable MD045 -->
<!-- markdownlint-disable MD032 -->
<!-- markdownlint-disable MD024 -->
<style>
  /* 強制網頁上所有元素（包括文字、段落、標題）都維持預設箭頭，不變一束 */
  body, body * {
    cursor: default;
  }
</style>

# AI outline graph

```mermaid
flowchart TD
    subgraph AI ["Artificial Intelligence"]
        direction TB
        subgraph ML ["Machine Learning"]
            direction TB
            subgraph NN ["Neural Networks"]
                direction TB
                subgraph DL ["Deep Learning (Multiple Layers)"]
                    direction TB

                    %% NLP Path
                    subgraph NLP_Domain ["Natural Language Processing (NLP)"]
                        subgraph RNN_Sub ["Recurrent Models"]
                            RNN["RNN"] --引入Gating Unit--> LSTM["LSTM"]
                        end
                        subgraph AB[Attention-Based]
                          TF["Transformer"]
                        end
                    end

                    %% CV Path
                    subgraph CV_Domain ["Computer Vision (CV)"]
                        CNN["CNN (Convolutional)"]
                    end

                    subgraph GenAI ["Generative AI"]
                        direction TB
                        LLM(("Large Language Models"))
                        GAN["GAN (Generative Adversarial Network)"]
                        DM["Diffusion Models"]
                    end
                    
                    %% Connections
                    RNN_Sub -.拋棄了遞歸架構，改用注意力機制.-> TF
                    TF --基礎架構--> LLM
                    TF -.DiT (Diffusion Transformer).-> DM
                    CNN --Spatial Representation Architecture--> GAN
                    CNN -.U-Net Backbone,特徵提取與重建.-> DM
                    AE["Autoencoder"] -.Latent Space Compression,壓縮至潛在空間.-> DM
                end
            end
        end
    end

    %% 註解說明
    Note["註:本圖側重於深度學習演進路徑"]
    Note --- AI

    %% Styles
    style AI fill:#4a90e2,stroke:#003366,color:#fff
    style ML fill:#8db5f8,stroke:#005ce6
    style NN fill:#b3d1ff,stroke:#005ce6
    style DL fill:#dae8fc,stroke:#005ce6
    style GenAI fill:#f5faff,stroke:#005ce6
    style NLP_Domain fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 5 5
    style CV_Domain fill:#f3e5f5,stroke:#4a148c,stroke-dasharray: 5 5
```

---

# AI model history timeline

![ai model history](model_history_timeline.svg)

---

# Definition: Generative AI

## 讓computer學會產生<u>複雜</u>而<u>有結構</u>的物件, 就是<font color="blue">Classification</font>, 因為Generative AI就是給一個未完成的句子, 去猜接下來接哪一個token

- 複雜 : 無限可能
- 有結構 : 由有限的<u>基本單位</u>(toekn)所構成
- 基本單位(token) :
  - 影像 : 是由<font color="blue">像素</font>構成的. 每個像素又由RGB三個子像素組成,而每個子像素的可能值範圍為0-255. 影像只考慮兩個維度: width, height
  - 聲音訊號 : 是由<font color="blue">取樣點</font>構成的. 遠看是聲音訊號,拉進後則可以看到一個一個的取樣點.一秒鐘有多少取樣點,取決於取樣率(sampling rate);比如, 16kHz的取樣率意味著一秒鐘有16000個取樣點. 而每個取樣點有多少種可能的數值,則取決於取樣解析度(bit resolution);常見的16-bit解析度有65,536種可能的數值
  - 影片 : 就是<font color="blue">一連串的圖片</font>, 每一張圖片又叫<font color="blue">Frame</font>. 需考慮三個維度: width, height, time

## Chinchilla Scaling-Laws

<a href="https://arxiv.org/pdf/2001.08361" title="Scaling Laws">Scaling Laws</a>:

- **核心思想**: 在固定訓練算力（Compute Budget）下，模型越大越好（優先增加參數量 $N$）。
- **盲點**: 導致早期模型（如 GPT-3 175B）訓練資料量（Tokens）相對不足，模型處於「未充分訓練（Under-trained）」狀態。

<a href="https://arxiv.org/pdf/2203.15556" title="Chinchilla Scaling Laws">Chinchilla Scaling Laws</a>:

- **核心思想**: 在固定訓練算力下，最佳策略是讓模型參數（$N$,Parameters）與訓練資料（$D$,Training Tokens）同步成長。
- **最佳比例**:$$D \approx 20N$$
- **行動方針**: Train Smaller Models Longer
- **現代延伸**: 現代模型（如 Llama 3）為了最佳化推理成本，通常採用「過度訓練（Over-training）」策略，將比例調至 $D \approx 50N \sim  200N+$, 具體取值因模型設計目標（延遲 vs 成本 vs 準確度）而異。

<a href="https://arxiv.org/pdf/2408.03314">Test-Time Scaling</a>:

- **核心公式**:$$Performance = f(\text{Pre-training Compute}, \text{Post-training Compute}, \text{Test-Time Compute})$$
- **核心思想**: 當預訓練算力與資料接近瓶頸，將算力向「推理期（Test-Time）」傾斜，讓模型在輸出前進行「自我反思與模擬」。
- **Test-Time Compute 實現路徑**:
  - **生成/搜索（Search）**: 思考鏈（CoT）、蒙地卡羅樹搜尋（MCTS）。
  - **驗證/修正（Verify）**: 內省（Self-Correction）、多路採樣投票（Majority Voting）。
- **代表模型**: OpenAI o1 / o3、DeepSeek-R1。

---

# Retrieval Augmented Generation(RAG)

---

# Context Engineering的核心目標

- 避免塞爆Context(把需要的放進去,不需要的清出來)

- 常用招數
  - Select
    - 挑選需要的內容, e.g. RAG, Tool RAG, Memory RAG
  - Compress
  - Multi-Agent

---

# Machine Learning $\approx$ looking for a function form data

## Machine Learning flowchart

```mermaid
flowchart TD
    subgraph PREP["Data Preparation"]
        A[Data] --> B["Transforms / Augmentation"]
        B --> C[Dataset]
        C --> D[DataLoader]
    end

    subgraph SETUP["Model & Training Setup"]
        E["Model (nn.Module)"]
        F[Loss Function]
        G[Optimizer]
        H["Scheduler<br>學習率動態調整"]
        G -.-> H
    end

    D --> E
    E --> F --> G

    subgraph LOOP["Training Loop"]
        I[Forward + Backward + Optimizer.step]
        I -.-> J["Logger / Visualization<br>TensorBoard / WandB"]
        H -.-> I
        I --> K[Validation]
        K -.-> L["Early Stopping /<br>Checkpoint 判斷"]
        L -->|"Patience not reached<br>Continue"| I
    end

    G --> I
    L -->|"Patience reached<br>Stop"| M[Save Best Model]

    subgraph DEPLOY["Inference / Deployment"]
        N["Load Best Model<br>model.load_state_dict()"]
        O["model.eval()<br>torch.no_grad()"]
        P[Inference]
        Q["Post-processing"]
    end

    M --> N --> O --> P --> Q

    style A fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style I fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style M fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style P fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

| Stage | Learning Type | Purpose | Data Source & Scale | Limitations | Key References | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-training** | Self-Supervised Learning | 學習語言結構、世界知識、常識與基礎代碼邏輯（學會「文字接龍」）。 | 海量網頁、圖書、學術論文、代碼庫。<br>**(10T–100T+ Tokens)** | 偏見與幻覺根源於此階段的訓練數據；後續階段只能緩解，無法根除。 | GPT-3、LLaMA | **Base Model**<br>➔ 知識淵博但無法正常對話，只會盲目預測下一個字。 |
| **Instruction Tuning / SFT** | Supervised Fine-Tuning | 轉變交互模式，讓模型理解「提示詞→回答」的結構，學會聽懂人話並執行指令。 | 人工撰寫或高質量 AI 生成的「高質量問答對 (Q&A Pairs)」。<br>**(數萬至數百萬條 Prompt-Response)** | Pre-training 帶入的偏見與幻覺並未被修正，只是「未被放大」；語氣可能仍粗糙。 | InstructGPT (Ouyang et al., 2022) | **SFT / Instruct Model**<br>➔ 成為合格的答題機器，能理解並執行指令，但仍缺乏價值觀對齊與推理深度。 |
| **Alignment** | RLHF / RLAIF | 賦予模型價值觀：優化說話語氣 (Vibe)、建立安全防線（拒絕有害請求）、減少幻覺。RLAIF 以強大 AI 代替人類打分，已成主流替代方案。 | 人類或強大 AI（RLAIF）對多個回答進行的偏好排序 (Preferences)。<br>**(數十萬條偏好數據)** | 過度 RLHF 可能導致模型過於保守或討好用戶（Sycophancy）。 | InstructGPT、Claude (Constitutional AI) | **Aligned Model**<br>➔ 安全、友善、有立場的 AI 助理，但複雜推理深度仍受限。 |
| **Reasoning Training** | Pure RL / Process Reward Model | 引入思維鏈 (CoT) 與自我糾錯能力，讓模型「先想後答」，大幅提升數學、代碼等複雜任務表現。此階段通常與 Alignment 分開進行。 | 針對數學/代碼等客觀正確答案的獎勵機制；以自動化驗證器提供獎勵信號。<br>**(自動化獎勵環境，無上限)** | 推理成本高（長 CoT）；在無驗證器的開放性問題上可能過度自信。 | o1、o3、DeepSeek-R1 (GRPO) | **Reasoning Model**<br>➔ 我們最終在 ChatGPT、Claude、DeepSeek 上使用的安全、好用、會思考的 AI 助理。 |

![mlthreephases](mlthreephases.png)

- <font color="green">Pretrained Models</font>
  - NLP
    - Encoder(分類、理解)
      - [BERT](https://arxiv.org/pdf/1810.04805) / RoBERTa / ALBERT / ELECTRA
    - Decoder(生成、對話)
      - GPT / LLaMA / OPT
    - Seq2Seq(翻譯、摘要)
      - T5/ BART
  - Computer Vision
    - Backbone(特徵提取，給下游用)
      - [CNN](https://arxiv.org/pdf/1511.08458)：[ResNet](https://arxiv.org/pdf/1512.03385)(CNN+Residual) / VGG / EfficientNet / Inception / ConvNeXt
      - Transformer：[ViT(Vision Transformer)](https://arxiv.org/pdf/2010.11929) / Swin Transformer / DINOv2
    - Semantic Segmentation(語義分割)
      - CNN-based：[U-Net](https://arxiv.org/pdf/1505.04597) / ResU-Net / DeepLab / SegNet
      - Transformer-based：SegFormer / SETR / Swin-UNET
    - Object Detection(框出物件)
      - One-stage（快）：[YOLO](https://arxiv.org/pdf/1506.02640)(pure CNN)
      - Two-stage（準）：Faster R-CNN(CNN backbone + RPN)
      - Transformer：DETR / RT-DETR
    - Instance Segmentation(精確輪廓)
      - Mask R-CNN/ YOLO-Seg / SAM

## 0. ML三步驟

![mlthreesteps](mlthreesteps.png)

## 1. prepare data

- split data into  
  i. **training data** &ensp;&emsp;&emsp; 98%  
  ii. **validation data** &emsp;&emsp; 1%  
  iii. **public test data** &emsp;0.5%  
  iv. **private test data**&emsp;0.5%

![data spliting](dataspliting.png)

- **Data Scarcity / Insufficient sample size**(data不夠)
  - **Data Augmentation**
    - **Image Data Augmentation**
      - **幾何變換**:
        - **翻轉 (Flipping)**:水平或垂直翻轉圖片。
        - **旋轉 (Rotation)**:將圖片隨機旋轉特定角度（如 ± 15°）。
        - **縮放 (Scaling) 與裁切 (Cropping)**:放大、縮小或隨機裁切圖片中的物件，強迫模型學習不同尺寸的特徵。
        - **平移 (Translation)**:將圖片沿著水平或垂直方向移動。
      - **色彩與像素變換**:
        - **顏色調整**:隨機改變亮度、對比度、飽和度或色調。
        - **雜訊注入**:加入高斯雜訊 (Gaussian Noise) 或椒鹽雜訊，模擬惡劣拍攝環境。
      - **影像混合 (Image Mixing)**:
        - **Mixup**:將兩張不同的圖片按隨機比例（λ）疊加混合。
        - **CutMix**:將一張圖片的一部分區域挖除，並用另一張圖片的內容填補。
    - **NLP Data Augmentation**
      - **詞彙替換**:使用同義詞詞庫（如 WordNet）將單字替換為同義詞。
      - **詞彙操作**:隨機對句子中的單字進行插入 (Insertion)、刪除 (Deletion) 或交換 (Swap)。
      - **遮罩語言模型 (Masked Language Model)**:利用像 BERT 等預訓練模型，將句中部分單字遮蔽 (Mask) 並讓模型自動預測填補，生成語意相同但寫法不同的新句子。
      - **回譯 (Back-Translation)**:將句子翻譯成另一種語言（例如中文翻成英文），再翻譯回原始語言，以獲得句型變化的資料。
    - **Audio Data Augmentation**
      - **時間平移 (Time Shifting)**:將音訊訊號往前或往後移動。
      - **時間拉伸 (Time Stretching)**:改變音訊的播放速度（加速或減速）而不改變音調。
      - **音調調整 (Pitch Shifting)**:改變音訊的音調高低。
      - **背景雜訊加入**:在語音中混入環境音（如咖啡廳噪音、街道聲），使模型適應真實環境。

## 2. set a model

- 根據domain knowledge
  - <font color="green">Simple Linear Regression</font>
    - $y = \beta_0 + \beta_1x + \epsilon$
      - 其中 $\beta_0$ 為截距，$\beta_1$ 為斜率，$\epsilon$ 為隨機誤差。
      - **$y, x$** is <font color="blue">feature</font>.**$\beta1$** is <font color="blue">weight</font>.**$\beta0$** is <font color="blue">bias</font>
      - **$\text{Parameters} = \text{Weights} + \text{Biases}$**
  - <font color="green">Multiple Linear Regression</font>
    - $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon$
  - <font color="green">Piecewise Linear Curves</font>
     - RELU(Rectified Linear Unit)
![piecewise linear curves](piecewiselinearcurves.png)
     - GELU(Gaussian Error Linear Unit)
     - Swish
     - SiLU(Sigmoid Linear Unit)
  - <font color="green">Classification</font>
    - <font color="green">Binary Classification</font>
      - Logistic Regression (Sigmoid Function)
        - Goal: predict probability of class 1
        - Formula:
          $$P(y=1|x)=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+\dots+\beta_nx_n)}}$$

        - Sigmoid Function:

          ![Sigmoid Function](Sigmoid-Activation-Function.png)

        - Output:
          - $0 \leq P(y=1|x) \leq 1$
          - Usually threshold = 0.5
            - $P(y=1|x)\geq0.5 \rightarrow y=1$
            - $P(y=1|x)<0.5 \rightarrow y=0$

        - Evaluation:
          - [confusion matrix](confusion_matrix.pdf)
    - <font color="green">Multi-class Classification</font>
      - Softmax Function

        - Goal: predict probability of each class

        - Formula:
          $$P(y=k|x)=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}$$

        - Softmax Function:

          ![Softmax](softmax.png)
  - <font color="green">Structured Learning</font>
    - **create** something with structure(e.g. image, document)

### compare regression and classification

![compare regression and classification](compare.png)

- $\sigma$ is called **activation function**
  - Regression(output is a "<font color="blue">scalar</font>")
    - $\sigma$ is Relu
  - Classification
    - $\sigma$ is sigmoid/softmax
- $\sigma(\mathbf{b} + \mathbf{W}\mathbf{x})$ is a **<font color="blue">neuron</font>**
- $f(\mathbf{x}) = b + \mathbf{c}^T \sigma(\mathbf{b} + \mathbf{W}\mathbf{x})$ is **<font color="blue">1-Hidden Layer Neural Network</font>**

## 3. set Cost function/Loss function

- Mean Squared Error, MSE
  - $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
    - $n$ : 訓練資料的總數 (Train Data Size / Batch Size).
    - $y_i$ : 第 $i$ 筆資料的真實標籤 (Ground Truth).
    - $\hat{y}_i$ : 模型對第 $i$ 筆資料的預測值 (Prediction)。
- Mean Absolute Error, MAE
  - $MAE = \frac{1}{n} \sum_{i=1}^{n} \left|(y_i - \hat{y}_i)\right|$
    - $n$ : 訓練資料的總數 (Train Data Size / Batch Size).
    - $y_i$ : 第 $i$ 筆資料的真實標籤 (Ground Truth).
    - $\hat{y}_i$ : 模型對第 $i$ 筆資料的預測值 (Prediction)。
- Cross-Entropy(classification)
  - if $y, \hat{y}$ are both probability distributions
  - $ Cross-Entropy=H(y, \hat{y}) = -\sum_{i=1}^{c} y_i \log(\hat{y}_i)$
    - $c$: The total number of classes or categories.
    - **Minimizing cross-entropy** is equivalent to **maximizing likelihood**
    - $y_i$: The probability of the $i$-th class in the true distribution (often a <font color="blue">one-hot encoded</font> vector where only one $y_i = 1$ and others are $0$).
    - $\hat{y}_i$: The probability of the $i$-th class predicted by the model (usually the output of a <font color="blue">Softmax function</font>($0 \le \hat{y}_i \le 1$ and $\sum \hat{y}_i = 1$.)).

## 4. set optimizer

- <font color="green">Gradient Descent(Vanilla Gradient Descent)</font>
  - $ \theta ^ {t+1} = \theta ^ {t} - \eta \cdot \nabla L(\theta ^ {t})$
    - $\nabla L(\theta ^ {t})$ 代表損失函數 $L$ 在當前參數 $\theta ^ {t}$ 位置的「斜率」或「坡度」
      - [Backpropagation](BP.pdf): an efficient way to compute $\frac{\partial L}{\partial w}$ in neural network
    - $\eta$(Eta) : The Learning Rate
    - 原始版本 : 每次更新要用「全部」訓練資料算一次 $\nabla L(\theta)$ → 稱為 <font color="green">Batch Gradient Descent</font>
      - 問題 : 資料量大時，算一次梯度非常慢，且記憶體吃不下
  - <font color="green">Stochastic Gradient Descent (SGD)</font>(<ins>SGD+Momentum 在 CNN 分類任務上常見，不代表 Adam 不適合</ins>)
    - 解決 Batch GD 太慢的問題 : 每次只用「一筆」資料（或一個 mini-batch）來估計梯度並更新參數
      - $ \theta ^ {t+1} = \theta ^ {t} - \eta \cdot \nabla L_i(\theta ^ {t})$
        - $L_i$ 是第 $i$ 筆(或一個 batch)資料算出的 loss
    - 優點 : 更新頻繁、速度快，且帶有隨機性(noisy)，反而有機會跳出 local minimum / saddle point
    - 缺點 : 更新方向震盪較大，收斂路徑不穩定，需要搭配 learning rate schedule 或 Momentum 才好收斂
    - 實務上常說的「SGD」多半是指 <font color="green">Mini-batch SGD</font>（一次用一小批資料，如 batch=32/128），純粹用單筆資料的版本很少見，因為效率太差
    - PyTorch 的 `optim.SGD` 實際上就是 mini-batch 版本，並且內建可選 momentum 參數（`momentum=0.9`）
  - 問題 : 只根據當下算出來的$g^t$來決定方向
    - 解法 : Gradient Descent + Optimizer
      - 根據$g^0,g^1,g^2,...,g^t$一起來決定方向(調整learning rate)
        - <font color="green">Adagrad</font>
          - for each dimension i : $\eta=\frac{\eta}{\sigma_i^t}$, $\sigma_i^t=\sqrt{\sum_{i=0}^{t} (g_i^t)^2}$
          - 問題 : 每一個維度權重相同
        - <font color="green">RMSProp</font>
          - 解決Adagrad : 最近算出來的gradient給比較大的影響
          - use $\alpha$
  - 可以用<font color="green"> Momentum </font>(下滑之物體會有動量繼續前行, 不會直接殺停)來繼續找
  global minimum用以脫離saddle point or local minimum
    - for each dimension i : $ \theta ^ {t+1} = \theta ^ {t} - \eta \cdot \color{purple}{m_i^t}$ , $\space$ $\color{purple}{m_i^t}$ = $\color{blue}{g_i^0}$ + $\color{green}{g_i^1}$ + $\color{orange}{g_i^2}$ +...+ $g^t$
  - <font color="green">Adam : RMSProp + Momentum</font>
![adam](adam.png)
  - 也可以用<font color="green">Learning Rate Scheduling</font>來調整learning rate
![Learning Rate Scheduling](learningratescheduling.png)
    - warp up : 給optimizer探索地形的機會, 因為剛進入一個新地圖, 不知道地圖有什麼, 設定一個大的learning rate, 讓參數亂跑, 可以大概知道地圖長什麼樣
  - Feature Scaling(夷平error surface)
  ![feature scaling](featurescaling.png)
    - accelerate gradient descent
  ![feature scaling](feature_scaling.webp)
    - Normalization
      - Batch Normalization
        - 大幅加速收斂: 允許模型使用較高的學習率（Learning Rate）來縮短訓練時間。
        - 防止梯度消失: 讓資料避開飽和區（如 Sigmoid, Tanh 函數的兩端），有效解決梯度消失問題。
        - 具備正則化 (Regularization) 效果: 減少模型對 Dropout 的依賴，降低過擬合（Overfitting）的風險。
  ![batch normalization](batchnormalization.png)
        - $\gamma$, $\beta$ are another network parameters, 另外再被learned出來的;因為normalization完後,$\tilde{z}^1,\tilde{z}^2,\tilde{z}^3,...$之平均為0, 可能會對模型產生限制,所以加上$\gamma$, $\beta$
        - $\gamma$ initialize to $[1, 1, \dots, 1]^T$
        - $\beta$ initialize to $[0, 0, \dots, 0]^T$
      - Layer Normalization
    - Standardizatoin

## 5.Train the model

- Initialization
  1. set random seed
  2. <font color="green">Kaiming Initialization</font>
- **<font color="blue">Use gradient descent to train the model, accelerating convergence to the minimum loss and yielding the optimal model(Find the best $\beta_0, \beta_1, \beta_2, \beta_3,...., \epsilon$).</font>** (Basically, we're now [<ins>***tuning hyperparameters***</ins>](common_hyperparameters.html))

- Sample Code

  ```python
  # Iterating over the dataset per batch
  for inputs, targets in dataloader:
      # 0. Move data to GPU/CPU per batch
      inputs, targets = inputs.to(device), targets.to(device)
      # 1. Clear gradients per batch
      optimizer.zero_grad()
      # 2. Forward pass
      outputs = model(inputs)
      # 3. Compute loss
      loss = criterion(outputs, targets)
      # 4. Backward pass (Backpropagation)
      loss.backward()
      # 5. Update model parameters per batch
      optimizer.step()
  ```

- tips when training
  - dropout(training 時把一些neuron or input丟掉, 當然, 相對應的weight也會丟掉)
<img src="dropout.png" width="50%">
  - early stopping
<img src="earlystopping.webp" width="40%">

- general guide

![general guide](generalguide.png)

## 6. use validation data to evaluate your model

- be aware of <font color="blue">overfitting</font>

## 7. use test data to test your model

- Public test data
  - be aware of <font color="blue">overfitting</font>
- Private test data
  - Used for a more reliable final evaluation.

## 8. Deploy the model for [inference](inference-hardware-precision-energy.pdf)

- Once the model performs well on the test data, **deploy it to an inference environment**.
- The deployed model can then receive new, unseen input and generate predictions.

> - **On-Premise & Cloud**: Deploy on local servers for strict data control, or leverage Cloud Platforms (e.g., AWS, GCP) for high scalability and managed infrastructure.
> - **Edge & Embedded**: Run models directly on user devices (smartphones, IoT) or microcontrollers via TinyML to achieve zero-latency and full offline capabilities.
> - **End-to-End Pipeline**: Integrate the model into an automated workflow to seamlessly transform raw data inputs into final real-time prediction outputs.

---

# Convolutional Neural Networks, CNN

<img src="cnn-1.png" width="60%">

**channel** : R, G, B

![cnn benefit](cnn-2.png)
不需Fully Connected Network, weight太龐大了, 用**Receptive Field**(多大自己定義可大可小, 不同receptive field can be overlapped), 而每一個neron只關心自己的receptive field發生的事情
<img src="receptivefield.png" width="60%">

鳥嘴可能在任何地方, 所以可以parameters sharing, called <ins>**Filter**</ins>(有幾個Filter, output就有幾個channel)
<img src="filter-1.png" width="60%"> <img src="filter-2.png" width="35%">

parameter sharing in details: 所以convolution不需要fully connected
<img src="cnn2017-1.png" width="50%"><img src="cnn2017-2.png" width="50%">

<ins>用Filter掃過一張圖片其實就是**convolution**</ins>

<img src="cnn_filter_animation_downsampling.gif">

## CNN Pipeline:

![cnn whole picture](cnn-3.png)

**Pooling** :  
e.g. 把奇數rows, columns拿掉, 讓圖變小, max pooling, average pooling,...

<img src="cnn-4.png" width="55%"><img src="maxpoolingaveragepooling.jpg" width="45%">

**Flatten** :  
把matrix變成column

**Fully Connected Feedforward Network** :
![fully connected feedforward network](fullyconnectfeedforwardnetwork.png)

$\sigma(\mathbf{b} + \mathbf{W}\mathbf{x})$ is a **<font color="blue">neuron</font>**

## Demo

<img src="cnndemo-1.png" width="80%">
<img src="cnndemo-2.png" width="80%">
<img src="cnndemo-3.png" width="80%">

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 基礎卷積區塊 (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D)
# =====================================================================
class ConvPoolBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_size: int = 2):
        super(ConvPoolBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2, # `//` is Floor Division
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # `->`Return Type Hinting
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# =====================================================================
# 2. 完整 CNN 模型 (含 Kaiming 權重初始化)
# =====================================================================
class CompleteCNNModel(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,          
        out_classes: int = 10,          
        channels_list: list = [32, 64, 128], 
        dropout: float = 0.2
    ):
        super(CompleteCNNModel, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList() # nn.ModuleList() is a container in PyTorch designed to hold submodules in a list.

        # 1. 動態堆疊 Conv-Pool 區塊
        curr_in = in_channels
        for out_ch in channels_list:
            self.layers.append(ConvPoolBlock(curr_in, out_ch))
            curr_in = out_ch

        # 2. 全域自適應平均池化 & 展平(adativeavgpool2d 不用手算長寬, 直接變成指定大小)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # [4, 128, 4, 4]->[4, 128, 1, 1] （把每個channel的 4*4 壓縮成 1*1 的平均值）
        self.flatten = nn.Flatten() # [4, 128, 1, 1]->[4, 128]

        # 3. 全連接預測頭
        last_channel = channels_list[-1] # [-1] is the last item in list
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_classes)
        )

        # 4. 執行權重初始化
        self._init_weights()

    def _init_weights(self):
        """遍歷所有子模組進行初始化：Conv/Linear 使用 Kaiming 初始化，BN 使用標準初值"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal 用於搭配 ReLU 激活函數
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BN 初始化：weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # `->`Return Type Hinting
        for layer in self.layers:
            x = layer(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# =====================================================================
# 3. 測試與驗證 (Demo Run)
# =====================================================================
if __name__ == '__main__':
    BATCH_SIZE = 4
    IN_CHANNELS = 3
    OUT_CLASSES = 10

    dummy_images = torch.randn(BATCH_SIZE, IN_CHANNELS, 32, 32) #(Batch_size, In_channels, Height, Width)

    model = CompleteCNNModel(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        channels_list=[32, 64, 128],
        dropout=0.2
    )
    model.eval()

    with torch.no_grad():
        out = model(dummy_images)

    print("=== 完成權重初始化的 CNN 測試成功 ===")
    print(f"輸出 Logits 形狀: {out.shape}")  # [4, 10]
```

## CNN output size

### strided convolution=subsampling=downsampling

$$O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$$

\(O\) : Output size
\(I\) : Input size
\(K\) : Filter size
\(P\) : Padding
\(S\) : Stride

$$\text{output Channel}=\text{Number of Filter}$$

<img src="cnn_filter_downsampling.png" width="90%">

\(O\) : 4
\(I\) : 6
\(K\) : 3
\(P\) : 0
\(S\) : 1

### Transposed Convolution=Deconvolution=up-convolution=upsampling

$$O = (I - 1) \times S - 2P + K + \text{Output Padding}$$

$$(通常我們把 \text{Output Padding} 設為 0)$$
\(O\) : Output size
\(I\) : Input size
\(K\) : Filter size
\(P\) : Padding
\(S\) : Stride

$$\text{output Channel}=\text{Number of Filter}$$

<img src="cnn_filter_upsampling.png" width="90%">

\(O\) : 3
\(I\) : 2
\(K\) : 2
\(P\) : 0
\(S\) : 1

---

# Graph Neural Networks, [GNN](GNN.pdf)

## GNN pipeline:

<img src="gnn_pipeline.svg" width="90%">

### one round / one layer of message passing

<img src="oneroundofmessagepassing.png" width="90%">

## GNN Roadmap:

<img src="gnnroadmap.png" width="90%">

## Spatial-based convolution's Terminology:

<img src="spatial-basedterminology.png" width="90%">

## Demo

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch

# =====================================================================
# 1. 重構後的 Message Passing 圖層 (符合 PyG 慣例 + Self-loops)
# =====================================================================
class NN_MessagePassingLayer(MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, aggr: str = 'mean'):
        # 1. 將 aggr 傳給 super() 註冊
        super(NN_MessagePassingLayer, self).__init__(aggr=aggr)
        # Message 函數：輸入為 concat(x_i, x_j)，維度為 2 * input_dim
        self.messageNN = nn.Linear(input_dim * 2, hidden_dim)
        # Update 函數：輸入為 concat(x_i, aggr_out)，維度為 input_dim + hidden_dim
        self.updateNN = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: # `->`Return Type Hinting
        # 2. 自動加入自環 (Self-loops)，確保訊息傳遞包含節點自身
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 3. propagate 不需要傳入 self.messageNN / updateNN
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor: # `->`Return Type Hinting
        # 直接存取 self.messageNN
        return F.relu(self.messageNN(torch.cat([x_i, x_j], dim=-1)))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor: # `->`Return Type Hinting
        # 直接存取 self.updateNN
        return self.updateNN(torch.cat([x, aggr_out], dim=-1))


# =====================================================================
# 2. 完整 GNN 模型 (支援 Mini-batching & 多層堆疊)
# =====================================================================
class CompleteGNNModel(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int, 
        out_channels: int, 
        num_layers: int = 2, 
        dropout: float = 0.2,
        task: str = 'node',            # 'node' 代表節點分類, 'graph' 代表圖分類
        norm_type: str = 'batch'       # 'batch', 'layer', 或 None
    ):
        super(CompleteGNNModel, self).__init__()
        self.task = task
        self.dropout = dropout
        self.norm_type = norm_type

        self.layers = nn.ModuleList() # nn.ModuleList() is a container in PyTorch designed to hold submodules in a list.
        self.norms = nn.ModuleList() # nn.ModuleList() is a container in PyTorch designed to hold submodules in a list.

        # 根據參數選擇 Norm 層
        norm_cls = None
        if norm_type == 'batch':
            norm_cls = nn.BatchNorm1d
        elif norm_type == 'layer':
            norm_cls = nn.LayerNorm

        # 第一層
        self.layers.append(NN_MessagePassingLayer(in_channels, hidden_dim, hidden_dim))
        self.norms.append(norm_cls(hidden_dim) if norm_cls is not None else nn.Identity())

        # 中間層堆疊
        for _ in range(num_layers - 1):
            self.layers.append(NN_MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim))
            self.norms.append(norm_cls(hidden_dim) if norm_cls is not None else nn.Identity())

        # 預測頭 (Prediction Head)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if norm_type == 'batch' else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels)
        )

        # 執行權重初始化 (Kaiming Normal Initialization)
        self._init_weights()

    def _init_weights(self):
        """遍歷所有子模組：包含 Linear、BatchNorm1d 與 LayerNorm"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor: # `->`Return Type Hinting
        # 1. 多層訊息傳遞
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. 圖分類任務處理解析 (支援 Mini-batch)
        if self.task == 'graph':
            if batch is None:
                # 若無傳入 batch，預設整張圖為單一 batch
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # 3. 分類器輸出
        return self.classifier(x)


# =====================================================================
# 3. 測試與驗證 (Demo Run)
# =====================================================================
if __name__ == '__main__':
    IN_DIM = 16
    HIDDEN_DIM = 32
    OUT_CLASSES = 3

    # 建立 2 張小圖作 Batch 測試
    g1 = Data(x=torch.randn(3, IN_DIM), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    g2 = Data(x=torch.randn(4, IN_DIM), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
    
    # 組合為 Mini-batch (PyG 會自動生成 batch 向量)
    batch_data = Batch.from_data_list([g1, g2])

    # 測試圖分類任務
    model = CompleteGNNModel(IN_DIM, HIDDEN_DIM, OUT_CLASSES, num_layers=2, task='graph', norm_type='batch')
    model.eval()

    with torch.no_grad():
        out = model(batch_data.x, batch_data.edge_index, batch_data.batch)

    print("=== Batch 測試成功 ===")
    print(f"Batch 圖數: 2")
    print(f"總節點數: {batch_data.x.size(0)}")
    print(f"輸出 Logits 形狀: {out.shape}")  # 應為 [2, 3]
```

---

# Self-Attention(取代RNN)

![input is a set of vectors](selfattention-1.png)

e.g. 文字處理, 聲音訊號(取一個範圍called window(通常為25ms), 描述成一個vector called frame, 為了描述一整段聲音訊號, 每次都把window往右移10ms, 取各個window's frame), graph(consider each **node** as a vector)

![tokenize](tokenize.png)
如何把一個詞彙表示成一個向量?

  1. **one-hot encoding**(向量長度跟世界上存在的詞彙數目一樣多), 但無法看出各個詞彙的相關程度
  2. **word embedding**(向量是有語意的資訊)(Unsupervised Learning(Self-Supervised Learning))
     1. Static Word Embedding
        1. 一個詞永遠對應一個固定向量，無法解決「一詞多義」問題
        2. Word2Vec, GloVe, FastText
     2. Contextualized Word Embedding
        1. 向量會根據「當前句子的上下文」動態改變，可精準處理一詞多義
        2. <font color="red">E</font>mbeddings from <font color="red">L</font>anguage <font color="red">Mo</font>dels (ELMO), <font color="red">B</font>idirectional <font color="red">E</font>ncoder <font color="red">R</font>epresentations from <font color="red">T</font>ransformers(BERT)

![output-1](output-1.png)
e.g. POS tagging,...

![output-2](output-2.png)
e.g. sentiment analysis,...

![output-3](output-3.png)
e.g. translation, speech recognition,...

<img src="selfattention-2.png" width="50%">
<img src="selfattention-2-1.png" width="45%">

self-attention會吃一整個sequence的資訊,input幾個vector, output幾個vector; self-attention也可以疊加很多次

<img src="selfattention-2-2.png" width="75%">

self-attention是這樣運作的

<img src="dotproduct.png" width="50%">

現今最常拿來計算self-attention之module就是**Dot-Product**
現今self-attention最常用的是<u>**Dot-Product**</u>, 也用在<u>**Transformer**</u>上

<img src="selfattention-3.png" width="85%">

$a^1$跟$a^2, a^3, a^4$個別算關聯性(attention score)

<img src="selfattention-4.png" width="85%">

實作上, $a^1$也會跟自己算關聯性(attention score), 然後再過一個soft-max(也可用其他的activation function $\sigma$)做normalization

<img src="selfattention-5.png" width="85%">

接下來, 根據$\alpha^\prime$去抽取出這個sequence裡面重要的資訊, 怎麼抽取重要資訊?
會把$a^1, a^2, a^3, a^4$每個vector乘上$W^v$得到$v^1, v^2, v^3, v^4$, 再把$v^1, v^2, v^3, v^4$去乘上已經算出來的attention score$\alpha_{1,1}', \alpha_{1,2}', \alpha_{1,3}', \alpha_{1,4}'$, 再相加
<ins>**所以如果$a^1, a^2$attention score很高($\alpha_{1,2}'很大$), 那weighted sum就很可能比較接近$v^2$**</ins>

<img src="selfattention-6.png" width="85%">

$b^1, b^2, b^3, b^4$是一次同時被計算出來的

<img src="selfattention-7.png" width="85%">

用矩陣乘法角度來看<!-- 橘色方塊 --><span style="background-color: #F4A27E; color: #000000; padding: 4px 8px; font-weight: bold;">qⁱ</span>, <!-- 土黃色方塊 --><span style="background-color: #FAD06C; color: #000000; padding: 4px 8px; font-weight: bold;">kⁱ</span>, <!-- 藍色方塊 --><span style="background-color: #85B7E2; color: #000000; padding: 4px 8px; font-weight: bold;">vⁱ</span>, $W^q,W^k,W^v$是被learned出來的

<img src="selfattention-8.png" width="85%">

用矩陣乘法來看attention score

<img src="selfattention-9.png" width="85%">

用矩陣乘法來看weighted sum

## Fianl Self-Attention

<img src="selfattention-10.png" width="85%">

## 進階版 Self-Attention: Multi-head Self-Attention

可以想成一般self-attention是用$q$去找相關的$k$,而相關這件事情, 有很多種不同的形式, 有很多種不同的定義, 所以也許不能只有一個$q$, 應該要有多個$q$, 不同的$q$負責不同的$k$

<img src="multiheadselfattention-1.png" width="85%">
<img src="multiheadselfattention-2.png" width="85%">

先把$a^i$乘上$W^q$得到$q^i$, 再把$q^i$乘上兩個不同的matrix $W^{q,1}, W^{q,2}$, 分別得到$q^{i,1}, q^{i,2}$; $k, v$也是相同作法
再去算weighted sum

<img src="multiheadselfattention-3.png" width="85%">

再讓$b^{i,1}, b^{i,2}$通過一個transform(matrix $W^o$)得到$b^i$, 再送到下一層去

## Self-Attention 少了位置資訊, 所以可以加上Positional Encoding

<img src="positionalencoding.png" width="45%">

## Self-Attention 不只用在NLP(Transforemr, BERT)上

<img src="selfattentionforspeech.png" width="60%">

Truncated Self-Attention: 不要看一整句話, 看一個小範圍就好

<img src="selfattentionforimage-1.png" width="70%">

每個pixel看成是一個3-dimension vector,整張圖片就是5*10個vectors

<img src="selfattentionforimage-2.png" width="70%">
<img src="selfattentionforimage-3.png" width="70%">

CNN可看作是簡化版self-attention
self-attention可看作是複雜版CNN
<img src="selfattentionforimage-4.png" width="70%">
<img src="selfattentionforimage-5.png" width="70%">
因為CNN是self-attention之subset, 所以資料量小時self-attention容易overfitting

<img src="selfattentionvsrnn.png" width="70%">

### RNN(Recurrent Neural Network) 與 Self-Attention 的差異

1. **RNN 的資訊傳遞方式**
   - 在一般 RNN 中，當前時間步只能直接利用前面時間步累積下來的 hidden state，因此本質上是依序從左到右處理。
   - 若要取得很久以前的資訊，必須透過一連串 hidden state 傳遞，容易發生 long-term dependency 問題。

2. **Bi-RNN(（Bidirectional Recurrent Neural Network)**
   - 雖然同時考慮左側與右側資訊，但遠距離資訊仍需透過多個時間步傳遞。
   - 例如最右邊的黃色 vector 想利用最左邊的藍色 vector，其資訊仍需經過許多 hidden state 才能傳到目標位置。
   - 因此長距離依賴仍然較難學習。

3. **Self-Attention**
   - 每個 token 都可以直接與所有 token 計算 attention score。
   - 最右邊的黃色 vector 若想參考最左邊的藍色 vector，只需透過 Query-Key 相似度即可直接建立連結。
   - 不需要經過中間多層 hidden state 的資訊傳遞，因此更容易捕捉長距離依賴。

4. **最大的差別：可否平行化（Parallelization）**
   - RNN 的計算依賴前一個時間步的 hidden state，因此必須依序計算：
     \[
     h_t = f(h_{t-1}, x_t)
     \]
   - Self-Attention 中，整個序列的 Query、Key、Value 可以一次計算完成，因此能夠充分利用 GPU 進行平行運算。
   - Transformer 訓練速度遠快於 RNN，這也是 Transformer 能取代 RNN 的最重要原因之一。

### <a href="https://colab.research.google.com/drive/1Xhg8lQ1miCrZh2bwZv1LgCbS4XHpH19b?usp=drive_link" title="HW4 Self-Attention Colab ">HW4	Self-Attention</a>

<font color="blue">Conformer</font> 就是 Transformer + CNN 的結合，專門為語音設計的，是目前語音辨識的主流架構。

Conformer = **Transformer + CNN** 的混合架構，2020年 Google 提出，專門為語音設計。

為什麼單用 Transformer 不夠？
Transformer 的 Self-Attention 可以看到**整段序列**的關係，但對**局部細節**不夠敏感。
語音有兩種重要的資訊:
**局部**:相鄰幾幀的音素變化（子音、母音的轉換）→ CNN 擅長
**全局**:整句話的韻律、說話者特徵 → Transformer 擅長

單用 Transformer 只抓全局，單用 CNN 只抓局部，Conformer 把兩個合在一起。

---

# Transformer(seq2seq model)

[![Transformer](Transformer.png)](https://arxiv.org/pdf/1706.03762)

Input a sequence, output a sequence, the output length is determined by model.
e.g. speech recognition, machine translation, speech translation, chatbot, sentiment analysis, syntactic parsing, Multi-label classification(an object can belong to multiple classes), object detection,...

```mermaid
flowchart LR
subgraph ModelTasks [Key Concept: Output length is determined by model]
    direction LR
    A([一段聲音]) -- speech recognition --> B[/辨識出的文字/]
    C([一個語言句子]) -- machine translation --> D[/另一個語言句子/]
    E([一段聲音]) -- speech translation --> F[/翻譯出的文字/]
    G([input]) -- chatbot --> H[/response/]
  end
```

```mermaid
flowchart LR
    %% 定義配色方案
    classDef io fill:#fff4dd,stroke:#d4a017,stroke-width:2px;
    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;

    subgraph seq2seq [Seq2Seq model architecture]
        direction LR
        style seq2seq fill:#fdfdfd,stroke:#999,stroke-dasharray: 5 5

        %% 節點定義 - 使用引號包裹文字以避免解析錯誤
        A(["Input sequence"])
        B(Encoder)
        C(Decoder)
        D(["Output sequence"])

        %% 連線
        A ==> B
        B ==>|Context Vector| C
        C ==> D
    end

    %% 應用樣式
    class A,D io
    class B,C core
```

## seq2seq encoder

<img src="seq2seqencoder-1.png" width="75%">
<img src="seq2seqencoder-2.png" width="75%">

一個Block在做的事情是好幾個layer在做的事情

## 原始Transformer encoder(BERT)

<img src="transformerencoder.png" width="70%">
<img src="realtransformerencoder.png" width="25%">

這是一個block的輸出

<span style="background-color: #FCD7D7; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Input Embedding</span> Embedding Look-up (learned)

```mermaid
flowchart LR
    A[input] -->|tokenize BPE| B[token ID]
    B -->|"lookup W[id]"| C["embedding vector e<br/>dim: d_model"]
    C -->|"× √(d_model)"| D[scaled embedding]
```

- Tokenize(byte-pair encoding)先切詞 → 查表轉成 embedding vector e
- Encoder/decoder 兩個 embedding layers 與 pre-softmax linear transformation **共享同一組 weight matrix** $W \in \mathbb{R}^{V \times d_{model}}$
- 輸出乘上 $\sqrt{d_{model}}$, $Scaled Embedding(x) = Embedding(x)\times\sqrt{d_{model}}$(推論: ​與 positional encoding 尺度對齊(其數值範圍固定在 [−1,1]),避免 positional encoding 的訊號被稀釋或蓋過)

<span style="background-color: #FFFFFF; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Positional Encoding</span> 用 sin/cos 固定公式產生，逐元素加到 Embedding 上

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- $pos$ : the position of the token in the sequence (0, 1, 2, ...)
- $i$ : the dimension index, $i \in [0, d_{model}/2)$
- $d_{model}$ : the dimension of the token embedding vectors
- Even dimensions use $\sin$, odd dimensions use $\cos$, paired two by two

<span style="background-color: #FADBB3; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Multi-Head Attention</span> 就是 Multi-Head self-attention

<span style="background-color: #F8F5C1; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Add & Norm</span> 就是 Residual + Layer Norm

<span style="background-color: #D2EDFA; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Feed Forward</span> 就是 Fully Connected Feed-Forward Network

$$\text{FFN}(x) = \underbrace{(\overbrace{\max(0, \, xW_1 + b_1)}^{\text{Linear1 + ReLU}})W_2 + b_2}_{\text{Linear2}}$$

<span style="font-size: 24px;">Nx</span> 就是有好幾個block

### Encoder Architecture Applications

1. Seq2Seq Generation
   1. **串接模組** : `Encoder + Decoder`
   2. **運作機制** : Encoder 提煉輸入序列的資訊，輸出 $K, V$ 向量供 Decoder 進行交叉注意力 (Cross-Attention) 計算。
   3. **代表架構** : Original Transformer, T5, BART
   4. **典型應用** :
      1. **Machine Translation** : 例如 英 $\rightarrow$ 中 翻譯
      2. **Summarization** : 長文壓縮成短摘要
2. Classification & Tagging
   1. **串接模組** : `Encoder + Linear Head (Classifier)`
   2. **運作機制** : 將 Encoder 產生的特徵向量傳入簡單的線性層 (nn.Linear)，進行機率映射與類別判斷。
   3. **代表架構** : BERT, RoBERTa, ELECTRA
   4. **典型應用** :
      1. **Sentiment Analysis** : 判斷文本為正評或負評
      2. **Text Classification** : 區分新聞主題（如:體育、政治、科技）
      3. **Named Entity Recognition (NER)** : 標記 Token 層級的實體類別(如:人名、地名、組織、時間、貨幣及產品等)
3. Semantic Search & Matching
   1. **串接模組** : `Encoder Only` $\rightarrow$ 輸出向量並存入 Vector Database
   2. **運作機制** : 直接提取句子的向量表示(Sentence Embedding)，透過餘弦相似度 (Cosine Similarity) 等幾何距離比對兩段文字的語意相關性。
   3. **代表技術** : Sentence-BERT, Dense Passage Retrieval (DPR)
   4. **典型應用** :
      1. **檢索增強生成 (RAG)** : 比對 User Query 與知識庫文件的相關度
      2. **語意搜尋 (Semantic Search)** : 突破傳統關鍵字匹配，實現「意思接近就能搜到」
4. **Computer Vision (CV) & Multimodal**
   1. **串接模組** : `Patch Embedding + Encoder + MLP Head`
   2. **運作機制** : 將圖片切割成固定大小的圖像塊 (Patches) 並壓平為 1D 序列，加上位置編碼 (Positional Encoding) 後送入 Encoder 提煉全域視覺特徵。
   3. **代表架構** : Vision Transformer (ViT), Swin Transformer, CLIP (Vision Encoder)
   4. **典型應用** :
      1. **Image Classification** : 圖像類別識別與物件分類
      2. **Cross-Modal Retrieval / Multimodal** : 將圖像與文字對齊至同一向量空間（如 CLIP），實現以圖搜圖或Text-Image Alignment

## Transformer decoder - Autoregressive decoder(最常見)

<img src="atvsnat.png" width="70%">
<img src="transformerdecoder-1.png" width="85%">
<img src="transformerdecoder-2.png" width="85%">

有可能出現error propagation, 就是一步錯步步錯

<img src="transformerdecoder-3.png" width="65%">
<img src="transformerdecoder-4.png" width="25%">

### <span style="background-color: #FADBB3; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Masked Multi-Head Self-Attention</span>

<img src="maskedselfattention-1.png" width="75%">

產生$b^1$只能考慮$a^1$的資訊
產生$b^2$只能考慮$a^1, a^2$的資訊
產生$b^3$只能考慮$a^1, a^2, a^3$的資訊
產生$b^4$只能考慮$a^1, a^2, a^3, a^4$的資訊

<img src="maskedselfattention-2.png" width="75%">

實際做的事情, 假設要產生$b^2$

**why masked??**
其實很直覺,因為decoder是一個一個的輸出,先有$a^1$再有$a^2$再有$a^3$再有$a^4$, 跟原始self-attention完全不一樣, 原始self-attention是$a^1, a^2, a^3, a^4$一起輸進去encoder裡, 因此跟encoder不一樣

### Cross attention

<img src="crossattention.png" width="65%">

<div style="display: flex; align-items: center; gap: 30px; background-color: #F8F9FA; padding: 15px 20px; border-radius: 8px; border: 1px solid #E0E0E0; max-width: fit-content; margin-left: 0; margin-right: auto; justify-content: flex-start;">
  <div style="text-align: left;">
    <div style="width: 32px; height: 32px; border: 4px solid #0055FF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #0055FF; background-color: white;">K</div>
    <div style="font-size: 12px; margin-top: 5px; color: #555; font-weight: bold;">Encoder</div>
  </div>

  <div style="text-align: left;">
    <div style="width: 32px; height: 32px; border: 4px solid #0055FF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #0055FF; background-color: white;">V</div>
    <div style="font-size: 12px; margin-top: 5px; color: #555; font-weight: bold;">Encoder</div>
  </div>

  <div style="text-align: left;">
    <div style="width: 32px; height: 32px; border: 4px solid #22C55E; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #22C55E; background-color: white;">Q</div>
    <div style="font-size: 12px; margin-top: 5px; color: #555; font-weight: bold;">Decoder</div>
  </div>
</div>

<img src="crossattention-1.png" width="45%">
<img src="crossattention-2.png" width="45%">

<!-- 橘色方塊 --><span style="background-color: #F4A27E; color: #000000; padding: 4px 8px; font-weight: bold;">q</span>, <!-- 土黃色方塊 --><span style="background-color: #FAD06C; color: #000000; padding: 4px 8px; font-weight: bold;">kⁱ</span>, <!-- 藍色方塊 --><span style="background-color: #85B7E2; color: #000000; padding: 4px 8px; font-weight: bold;">vⁱ</span>, 是被<span style="background-color: #85B7E2; color: #000000; padding: 4px 8px; font-weight: bold; #000000; display: inline-block;">W<sup>q</sup></span> <span style="background-color: #85B7E2; color: #000000; padding: 4px 8px; font-weight: bold; #000000; display: inline-block;">W<sup>k</sup></span> <span style="background-color: #85B7E2; color: #000000; padding: 4px 8px; font-weight: bold; #000000; display: inline-block;">W<sup>v</sup></span>乘出來的

### <span style="background-color: #E2E2F6; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Linear</span> to <span style="background-color: #E2F0D9; color: #000000; padding: 4px 12px; border-radius: 6px; font-weight: bold; border: 2px solid #000000;">Softmax</span> :

$$\text{Output}(x) = \underbrace{\text{Softmax}(\overbrace{xW + b}^{\text{\color{#6B62B2}Linear (Logits)}})}_{\text{\color{#3E7A44}Softmax (Probabilities)}}$$

## Train

<img src="transformertraining-1.png" width="90%">
<img src="transformertraining-2.png" width="90%">

訓練時會把正確答案當成decoder之輸入called **teacher forcing**

## Demo

```python
import torch
import torch.nn as nn
import math

# --- 1. 原論文經典正弦位置編碼 (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x Shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# --- 2. 完整 Seq2Seq Transformer ---
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
        num_encoder_blocks=6, num_decoder_blocks=6, pad_idx=0): 
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding & Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Block
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_blocks,
            num_decoder_layers=num_decoder_blocks,
            batch_first=True
        )
        
        # Final Output Head
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # 1. 詞嵌入 + 縮放 (論文要求乘以 sqrt(d_model)) + 位置編碼
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 2. 丟入 Transformer
        out = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 3. 輸出映射
        logits = self.fc_out(out)
        return logits

        # 不需要過nn.Softmax(), softmax在nn.CrossEntropyLoss()裡面了
        # CrossEntropyLoss = LogSoftmax + NLLLoss (Negative Log Likelihood)

# --- 3. 測試執行範例 ---
pad_idx = 0
model = Seq2SeqTransformer(src_vocab_size=10000, tgt_vocab_size=10000, pad_idx=pad_idx)

src = torch.randint(1, 10000, (32, 10)) # 假設 0 是 pad_idx
tgt = torch.randint(1, 10000, (32, 12))

# 建立 Masks
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)) # Look-ahead mask
src_padding_mask = (src == pad_idx)                                     # Pad mask
tgt_padding_mask = (tgt == pad_idx)

# 前向傳播
output = model(
    src, tgt, 
    tgt_mask=tgt_mask, 
    src_padding_mask=src_padding_mask, 
    tgt_padding_mask=tgt_padding_mask
)

print("Output Shape:", output.shape) 
# -> torch.Size([32, 12, 10000])
```

---

# Generative Adversarial Network(GAN)

## Generator

## Discriminator

---

# BERT

BERT is a self-supervised learning(learn from **unlabeled** data), ChatGPT, Gemini, Claude, Llama, and Grok are also self-supervised learning  
BERT一般用在自然語言處理上,當然語音與圖片也可當作seq2seq處理

<img src="bert.png" width="80%">

HOW TO TRAIN BERT?(有兩個任務要學)(it is **Unsupervised**)
<img src="howtotrainbert-1.png" width="80%">
<img src="howtotrainbert-2.png" width="80%">

sentence1與sentence2是否相接

所以BERT學會做填空題, 那要如何使用BERT?  
BERT就像是幹細胞

<img src="howtousebert.png" width="80%">

they are **Semi-Supervised**

<img src="howtousebertcase1.png" width="45%">
<img src="howtousebertcase2.png" width="45%">
<img src="howtousebertcase3.png" width="45%">
<img src="howtousebertcase4.png" width="45%">

---

# Explainable ML

<img src="whyexplainableml.png">

- Goal of Explainable ML
  - <font color="blue">Make people comfortable</font>

- **saliency map**
  - A **saliency map** is a **visual representation that highlights the most prominent or influential regions in a piece of data**, such as pixels in an image or tokens in a text. Its primary goal is to identify exactly which features most strongly influenced a specific outcome.
<img src="smoothgrad.png" widht=45%>
why saliency map? ans:make people comfortable

---

# Reinforcement Learning

what is Reinforcement Learning?
ans:<font color="green">It is still a machine learning method used when labeled data is unavailable; the model learns through rewards and penalties from its actions.</font>
![reinforcement learning](reinforcementlearning.png)
e.g.Space invader, Alpha go
<img src="spaceinvader.png" width="80%">
<img src="alphago.png" width="80%">

從遊戲開始到結束稱為一個<ins>**episode**</ins>, 每過一個episode算一次<ins>total reward</ins>(return)

<img src="policygradient.png">
The <font color="blue">trajectory</font> is defined as $\tau=\{s_1,a_1,s_2,a_2,\dots\}$

train on reinforcement learning 很卡的一個地方是:每次更新一次參數就要蒐集一次資料, called <font color="blue">on-policy</font>(the <font color="green">actor to train</font> and the <font color="orange">actor for interacting</font> is the same)

---

# Life Long Learning

學完task1 再學task2, 可能會忘記task1學過的東西

<img src="catastrophicforget.png" width="60%">

跟Transfer Learning不一樣,
**Transfer Learning** is<font color="blue"> <ins>I can do task2 because I have learned task1.</ins></font>
**Life Long Learning** is <font color="blue"><ins>even though i have learned task2, I do not forget task1.</ins></font>

## Evaluate life long learning model(3 types)

<img src="lllevaluation.png" width="70%">

<img src="lllbackwardtransfer.png" width="70%">

backward transfer是看model的遺忘程度

<img src="lllforwardtransfer.png" width="70%">

forward transfer是還未看到$T_T$, 只看到$T_1$到$T_T-1$時, 看model學到甚麼程度

## 解決Catastrophic Forgetting

### Selective Synaptic Plasticity

<img src="SelectiveSynapticPlasticity-1.png" width="80%">
<img src="SelectiveSynapticPlasticity-2.png" width="80%">
<img src="SelectiveSynapticPlasticity-3.png" width="80%">

$b_i$是人為設定的
<img src="SelectiveSynapticPlasticity-4.png" width="80%">

---

# Network Compression

why Network Compression?
<img src="networkcompression.png" width="70%">

- 方法
  - Network Pruning
    - weight pruning(不推薦, 因為矩陣乘法還是沒有變小, 只是把weight改成0而已)
    - neuron pruning(easy to implement, easy to speedup)
  - Knowledge Distillation
  - Parameter Quantization
  - Architecture Design(以CNN為例,
    - Depthwise Separable Convolution
      - Depthwise Convolution + Pointwise Convolution )
  - Dynamic Computation(希望network可以自由地調整自己需要的運算量)

---

# Meta Learning

what does "Meta" mean?
<font color="green">meta-X is <ins>X about X</ins></font>, so Meta Learning is <font color="blue"><ins>Learning about Learning</ins></font>.

<img src="metalearning.png" width="80%">
<img src="metalearninggoal.png" width="80%">

## meta learning steps

<img src="metalearningstep1.png" width="80%">
<img src="metalearningstep2-1.png" width="80%">
<img src="metalearningstep2-2.png" width="80%">
<img src="metalearningstep2-3.png" width="80%">
<img src="metalearningstep2-4.png" width="80%">

<img src="metalearningstep2-5.png" width="60%">

跟一般machine learning不太一樣, meta learning是用testing examples來算loss

<img src="metalearningstep3.png" width="80%">

## compare machine learning & meta learning

<img src="compare-1.png" width="80%">
<img src="compare-2.png" width="80%">
<img src="compare-3.png" width="80%">
<img src="compare-4.png" width="80%">
<img src="compare-5findminloss.png" width="60%">
