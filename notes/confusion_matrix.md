<!-- markdownlint-disable MD036 -->
# 分類模型與電腦視覺評估指標筆記

**Classification & Computer Vision Metrics｜Medical Image Segmentation**

> 目前研究方向為 **SegResNet / nnU-Net / 3D Medical Imaging**，請特別留意「四、醫學影像 Segmentation 專用指標」。

---

## 30 秒總覽（先建立地圖，再看細節）

```text
分類指標 (混淆矩陣)          分割指標 (Segmentation)
  ├─ 排序能力：AUC              ├─ Region  重疊得好不好？  Dice / IoU
  ├─ 抓病人：  Sensitivity      ├─ Boundary 邊界準不準？   HD / HD95 / ASSD / NSD
  ├─ 排健康：  Specificity      └─ Volume  體積接不接近？  VS
  ├─ 報警準：  PPV (Precision)
  └─ 排除準：  NPV
```

看到論文數字時的直覺反應：

| 看到這個 | 立刻想 |
| --- | --- |
| Dice / IoU 高 | 重疊得好 |
| HD95 / ASSD 小 | 邊界誤差小 |
| NSD 高 | surface 落在容許範圍內的比例高 |

**最基本組合：** $\boxed{Dice + HD95}$　**更完整：** $\boxed{Dice + HD95 + ASSD/NSD}$

---

## 零、混淆矩陣基礎（Confusion Matrix）

### 1. 表格結構

所有分類指標都是從這張 2×2 表格衍生出來的，是最基本的視覺錨點：

| | 預測 Positive（陽性） | 預測 Negative（陰性） |
| --- | --- | --- |
| **實際 Positive** | TP（True Positive，真陽） | FN（False Negative，假陰／類似 Type II Error） |
| **實際 Negative** | FP（False Positive，假陽／類似 Type I Error） | TN（True Negative，真陰） |

> **判讀口訣：** 第一個字看對錯（True = 猜對，False = 猜錯）；第二個字看模型猜什麼（Positive = 猜正類，Negative = 猜負類）。
> 例如 FN = 猜錯（False）+ 猜成陰性（Negative）→ 其實是陽性，但模型說是陰性，也就是「漏診」。

### 2. 實務觀念：門檻調校與正規化

- **分類閾值（Threshold）**：混淆矩陣的數值會隨判定門檻改變。降低門檻時（如由 $0.5$ 降至 $0.2$），通常會使更多樣本被判為 Positive，因此 **TP 與 FP 傾向增加，FN 與 TN 傾向減少**（提升 Sensitivity，降低 Specificity）。
- **正規化混淆矩陣（Normalized Confusion Matrix）**：當正負樣本數量懸殊時，將矩陣轉換為比例/百分比（0.0 ~ 1.0），有助於更客觀地比較不同類別的辨識表現。

---

## 一、分類評估指標（Classification Metrics）

### 1. AUC（Area Under the ROC Curve）

衡量模型在不同 Threshold 下的**排序與區分能力**。

- 本質：隨機抽一個正樣本與一個負樣本，模型給正樣本較高分數的機率
- `AUC = 1.0` 完美　`AUC = 0.5` 等同隨機猜測　`AUC < 0.5` 預測方向相反
- ⚠️ 極度不平衡資料下，AUC 仍可能過度樂觀，需搭配 Sensitivity、PPV 一起看

### 2. Accuracy（準確率）

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

最直覺，但在**資料極度不平衡**時會失真（例如 99 個健康、1 個生病，全猜健康也有 99% Accuracy）。

### 3. Sensitivity（敏感度 / Recall / TPR）

$$
\text{Sensitivity} = \frac{TP}{TP + FN}
$$

站在「真實正類」角度：有多少真正有病的人被抓出來。

### 4. Specificity（特異度 / TNR）

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

站在「真實負類」角度：有多少真正健康的人被正確判斷為陰性。

> **好記口訣 — SnNout / SpPin：**
>
> - **SnNout**：Sensitivity 高的檢測，**N**egative 結果可以放心排**out**（排除疾病）
> - **SpPin**：Specificity 高的檢測，**P**ositive 結果可以放心確**in**（確診疾病）
>
> 對應到情境：癌症篩檢要「寧可錯殺不可放過」→ 要高 Sensitivity（SnNout）；
> 確診用的檢測要「報陽性就是真的有」→ 要高 Specificity（SpPin）。

### 5. PPV（Precision / 陽性預測值）

$$
\text{PPV} = \frac{TP}{TP + FP}
$$

站在「模型預測為陽性」的角度：警報有多準。受盛行率（Prevalence）影響很大。

### 6. NPV（陰性預測值）

$$
\text{NPV} = \frac{TN}{TN + FN}
$$

站在「模型預測為陰性」的角度：告訴你「沒問題」時，你可以有多安心。

### 7. F1-Score 與 Dice Coefficient

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
\qquad
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}
$$

- **F1 的本質**：模型抓到的正類有多準（Precision）、又抓到多少該抓的（Recall），這兩者的平衡好不好？用調和平均計算，所以只要有一項很低，整體分數就會被拉低。
- **F1 與 Dice 的關係**：在**二元分割且以 pixel 為單位**時，F1 與 Dice 數學上等價（後面「四、1」的 Dice 就是同一個東西，只是換到分割脈絡再強調一次）——但這是特殊情境下的巧合，F1 本身不是為了量「重疊」而生的
- 多類別分割通常改用 per-class Dice 再取 mean Dice

### 8. 其他常用指標

| 指標 | 公式 | 說明 |
| ------ | ------ | ------ |
| **FPR** | $\dfrac{FP}{FP+TN} = 1 - \text{Specificity}$ | 假陽性率（誤報率），ROC 的 X 軸 |
| **FNR** | $\dfrac{FN}{TP+FN} = 1 - \text{Sensitivity}$ | 假陰性率（漏報率） |
| **MCC** | $\dfrac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | 範圍 -1 ~ +1，極度不平衡時較穩健 |

---

## 二、電腦視覺常見指標（Detection & Segmentation）

### 1. IoU（Intersection over Union / Jaccard Index）

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}
$$

預測區域與真實區域的重疊程度，Object Detection 與 Segmentation 都常用。

### 2. Dice 與 IoU 的換算

$$
\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}} \qquad
\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}
$$

Dice 永遠 ≥ IoU（同一個重疊程度，Dice 的數字看起來比較「好看」）。

### 3. PR Curve、AP、mAP

- **PR Curve**：Recall 為 X 軸、Precision 為 Y 軸；在正類稀少的情況下，通常比 ROC Curve 更能反映模型對少數類的辨識能力
- **AP**：單一類別的 PR 曲線面積；**mAP** = 所有類別 AP 的平均

$$
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} AP_i
$$

- `AP@0.5`：IoU 門檻 = 0.5　`AP@0.5:0.95`：IoU 從 0.5 到 0.95、間隔 0.05 平均（COCO 標準）

### 4. PR-AUC

Precision-Recall Curve 下的面積。在**正類非常稀少**的情況下，通常比 ROC-AUC 更能反映模型對少數類的真實辨識能力。

---

## 三、錯誤類型（Type I & Type II）

| 錯誤 | 對應 | 意義 | 代價 |
| ------ | ------ | ------ | ------ |
| **FP**（假陽性） | 類似 Type I Error | 沒事報有事 | 誤診、假警報 |
| **FN**（假陰性） | 類似 Type II Error | 有事報沒事 | 漏診（通常更嚴重） |

> 統計假設檢定的 Type I / II 與分類的 FP / FN 只是**結構相似的類比**，並非完全等價。

---

## 四、醫學影像 Segmentation 專用指標（重點）

```text
                Segmentation Metrics
                       │
      ┌────────────────┼────────────────┐
      │                │                │
   Region           Boundary          Volume
  重疊得好不好？      邊界準不準？      體積接不接近？
      │                │                │
  Dice / IoU     HD / HD95 / ASSD      VS
                       │
                      NSD
```

### 1. Region 指標

**Dice**（與「一、7」同一個定義，這裡是分割脈絡下的主角）

$$
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2TP}{2TP + FP + FN}
$$

範圍 0~1，越高越好。回答：「Prediction 和 Ground Truth 重疊得有多好？」

**IoU**（同「二、1」）

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|}
$$

### 🔢 一個小範例，把數字釘進腦子裡

假設某張切片標註腫瘤共 100 個 pixel（Ground Truth），模型預測出 90 個 pixel，
其中兩者重疊 80 個 pixel：

| 值 | 數字 |
| --- | --- |
| $\lvert P \cap G \rvert$（重疊） | 80 |
| $\lvert P \rvert$（預測面積） | 90 |
| $\lvert G \rvert$（真實面積） | 100 |
| TP / FP / FN | 80 / 10 / 20 |

$$
\text{Dice} = \frac{2 \times 80}{90 + 100} = 0.842
\qquad
\text{IoU} = \frac{80}{90+100-80} = 0.727
\qquad
\text{Sensitivity} = \frac{80}{100} = 0.80
\qquad
\text{PPV} = \frac{80}{90} = 0.889
$$

代入 Dice/IoU 換算式驗證：$\dfrac{2 \times 0.727}{1+0.727} = 0.842$ ✓ 對得起來。

### 2. Boundary 指標

**Hausdorff Distance（HD）**

$$
h(A,B) = \max_{a \in A} \min_{b \in B} d(a,b)
\qquad
\text{HD}(A,B) = \max \big\{ h(A,B),\ h(B,A) \big\}
$$

衡量兩個邊界「最糟糕可以差多遠」，對單一 outlier 非常敏感。

**HD95（強烈建議使用）**：用表面距離分佈的 **95th percentile**，比原始 HD 穩定得多。
回答：「大部分邊界與 Ground Truth 大約差多遠？」單位通常是 **mm**（讀 paper 一定要確認單位）。

**ASSD / ASD（Average Symmetric Surface Distance）**：平均雙向表面距離，比 HD 穩定、比 HD95 更看整體平均。

**NSD（Normalized Surface Dice）**：有多少比例的 surface 落在指定 tolerance（例如 2mm）內。必須搭配 tolerance 一起解讀。

### 3. Volume 指標

**Volumetric Similarity（VS）**

$$
\text{VS} = 1 - \frac{|V_P - V_G|}{V_P + V_G}
$$

看預測體積與真實體積有多接近。⚠️ 不同論文定義可能略有差異，讀 paper 時留意作者自訂的版本。

### 指標互補關係一覽

| 類型 | 指標 | 核心問題 | 趨勢 |
| ------ | ------ | ---------- | ------ |
| Region | Dice / IoU | 重疊多少？ | ↑ 越好 |
| Boundary | HD | 最差邊界距離？ | ↓ 越好 |
| Boundary | HD95 | 95% 邊界距離？ | ↓ 越好 |
| Boundary | ASSD | 平均邊界距離？ | ↓ 越好 |
| Boundary | NSD | Surface 有多少在 tolerance 內？ | ↑ 越好 |
| Volume | VS | 體積有多接近？ | ↑ 越好 |

---

## 五、ROC 曲線快速理解

- **X 軸**：FPR（假陽性率）→ 越低越好
- **Y 軸**：TPR / Sensitivity → 越高越好
- 曲線越靠近左上角 (0,1) → 模型越好；對角線 → 隨機猜測

降低 Threshold 時：Sensitivity ↑、Recall ↑，但 Specificity ↓、Precision 通常下降。

---

## 六、速查表

| 指標 | 觀察角度 | 適用情境 |
| ------ | ---------- | ---------- |
| **AUC** | 排序與區分能力 | 跨模型比較 |
| **Accuracy** | 全局正確率 | 資料平衡時 |
| **Sensitivity** | 真實患者抓出率（SnNout） | 醫療篩檢（不可漏） |
| **Specificity** | 真實健康者辨識率（SpPin） | 減少假警報 |
| **PPV (Precision)** | 模型警報準確度 | 誤報代價高時 |
| **NPV** | 陰性結果可靠度 | 排除疾病時 |
| **F1 / Dice** | Precision 與 Recall 平衡 | 不平衡資料、分割 |
| **IoU** | 區域重疊程度 | Detection、Segmentation |
| **HD95** | 邊界品質 | 醫學影像分割 |
| **MCC** | 四個象限綜合 | 極度不平衡資料 |

---

## 七、目前研究方向最重要的組合

如果主要做 **Medical Image Segmentation / SegResNet / nnU-Net**：

$$
\boxed{Dice + IoU + HD95 + ASSD + NSD}
$$

再搭配：

$$
\boxed{Sensitivity + Precision + Specificity}
$$

就能建立一套完整且實用的評估觀念。
