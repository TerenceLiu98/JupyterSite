
# Use Python in Advanced Statistics

*這個學期剛剛學完了概率論與數理統計，這好趁這個機會複習一下並複習一下 `python`*
<br>




## Chapter One Probability 

### 隨機試驗與樣本空間（Random Experiment and Sample Space）

#### 隨機試驗
隨機試驗是概率論中一個基本的概念。概括的講，在概率論中把符合下面三個特點的試驗叫做隨機試驗：
* 可以在向空的條件下重複進行；
* 每次試驗的可能結果不只一個，並且事先明確試驗的所有可能結果；
* 進行一次試驗之前不能確定哪一個結果會出現。

隨機試驗有很多種，例如常出現的擲骰子，摸球，射擊等。所有的隨機試驗的結果可以分為兩類來表示：<br>
* 數量化表示：射擊命中的次數，商場每個小時的客流量，每天經過某個收費站的車輛等，這個結果本事就是數字；
* 非數量化表示：拋硬幣的結果（正面/反面），化驗的結果（陽性/陰性）等，這些結果是定型的，非數量化的。但是可以用示性函數來表示，例如可以規定正面（陽性）為$1$，反面為$0$，這樣就可以實現了非數量化結果的數量化。

#### 樣本空間（Sample Space）：

* 隨機試驗的所有可能結果構成的集合。一般即為$S$（capital S）；
* $S$ 中的元素$e$稱為樣本點（也可以叫基本事件）；
* 事件是樣本空間的子集，同樣是一個集合。

#### 事件的關係
* 事件的包含：$A \subseteq B$;
* 事件的相等：$A = B$;
* 互斥事件（互不相容事件）：不能同時出現；
* 事件的和（並）：$A cup B$
* 事件的差： $A - B$,$A$發生，$B$不發生；
* 對立事件（逆事件）：互斥，必須出現其中一個。

<br>
事件的運算性質就是集合的性質

### 頻率和概率

#### 頻率：

頻率是指$0～1$之間的一個實數，在大量重複試驗的基礎上給出了隨機事件發生可能性的估計。<br>
概率的穩定性：在充分多次試驗中，事件的頻率總在一個定值附近擺動，而且，試驗次數越多擺動越小。這個性質叫做頻率的穩定性。

#### 概率：

概率的統計性定義：當試驗次數增加時，隨機時間$A$發生的頻率的穩定值為$p$就稱為概率。記為$P(A) = P$<br>
概率的公理化定義：設隨機試驗對於的樣本空間為$S$。對每一個事件$A$，定義為$P(A)$，滿足：<br>
1. 非負性：$P(A) \geq 0$;
2. 規範性：$P(S) = 1;
3. 可列可加性：$A_1, A_2, \dots 兩兩互斥，及$A_iA_j = \oslash, i \neq j$則 $P(\cup A_i) = \sum P(A_i)$ 

#### 條件概率（Conditional Probability）:
$P(A|B)$表示在事件$B$發生的條件下，事件$A$發生的概率，相當於$A$在$B$所佔的比例。此時，樣本空間從原來的完整樣本空間$S$縮小到了$B$，由於有了條件的約束（事件$B$），使得原本的樣本空間減少了。

下面我們可以通過韋恩圖做示例：<br>
plot one：條件概率的樣本空間；<br>
plot two：條件概率應如何計算


```python
from matplotlib import pyplot as plt
import numpy as np
import sympy


from matplotlib_venn import venn3, venn3_circles
plt.figure(figsize=(4,4))
v = venn2(subsets=(2,2,1), set_labels = ('A', 'B'))



plt.title("Sample Venn diagram - plot one")
plt.annotate('P(AB)', xy=v.get_label_by_id('11').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
plt.show()

```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_8_0.png)



```python
from matplotlib import pyplot as plt
import numpy as np
import sympy


from matplotlib_venn import venn3, venn3_circles
plt.figure(figsize=(4,4))
v = venn2(subsets=(2,2,1), set_labels = ('A', 'B'))

c = venn2_circles(subsets=(2, 2, 1), linestyle='dashed')
c[0].set_lw(1.0)
c[0].set_ls('dotted')
plt.title("Sample Venn diagram")
plt.annotate('P(AB)', xy=v.get_label_by_id('11').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_9_0.png)


$P(B|A) = \frac{P(AB)}{P(A)}$<br>
$P(A|B) = \frac{P(AB)}{P(B)}$<br>

例題：一個家庭中有兩個小孩，已知至少一個是女孩，問兩個都是女孩的概率是多少？（假設生男生女是等可能的）<br>
**解**：由題意可得：樣本空間為
    
    S = {(兄,弟), (兄,妹),(姐,弟),(姐,妹)}<br>
    B = {(兄,妹), (姐,弟),(姐,妹)}<br>
    A = {(姐,妹)}<br>
    
由於，事件 $B$ 已經發生了，所以這時試驗的所有可能只有三種，而事件 $A$ 包含的基本事件只占其中的一種，所以有：$P(A|B) = \frac{1}{3}$<br>
即，在已知至少一個是女孩的請卡滾下，兩個都是女孩的概率為$\frac{1}{3}$。在這個例子中，如果不知道事件 $B$ 發生，則事件 $A$ 發生的概率為 $P(A) = \frac{1}{4}$ 這裡的 $P(A) \neq P(A|B)$，其中的原因在於事件 $B$ 的發生改變了樣本空間，使它由原來的 $S$ 縮減為新的樣本空間 $S_B = B$ 

#### 隨機變量（Random Variable）

在幾乎所有教材裡，介紹概率論都是從事件和樣本空間說起的，但是後面的概率論都是圍繞著隨機變量展開的。可以說前面的事件和樣本空間都是引子，引出了隨機變量這個概率論的核心概念。後面的統計學是建立在概率論的理論基礎之上的，因此可以說理解隨機變量這個概念是學習和運用概率論與數理統計的關鍵。<br>
**隨機變量**：<br>

* 首先這是一個變量，變量與常數相對，也就是說其取值是不明確的，其實隨機變量的整個取值範圍就是前面說的樣本空間；
* 其次這個量是隨機的，也就是說它的去職代有不確定性，讓然是在樣本空間這個範圍內的。


**定義：**
> 設隨機試驗的樣本空間是 $S$ ，若對 $S$ 中的每一個樣本點 $e$ ，都有唯一的實數值 $X(e)$ 為隨機變量，間記為 $X$

隨機變量的定義並不複雜，但是理解起來去不是這麼直觀。

* 首先，隨即變量與之前定義的事件是有關係的，因為每個樣本點本身就是一個基本事件；
* 在前面隨機試驗結果的表示中提到，無論是數量化的結果還是非數量化的結果，即不管試驗結果是否與數值有關，都可以引入變量，使試驗結果與數建立對應關係；
* 隨機變量本質上是一種函數，其目的就是建立試驗結果（樣本中的點，同基本事件$e$）與實數之間的對應關係（例如將“正面”影射為$1$，“反面”影射為$0$）；
* 自變量為基本事件$e$，定義域為樣本空間$S$，值域為某個實數集合，多個自變量可以對應同一個函數值，但不允許一個自變量對應多個函數值；
* 隨機變量$X$取某個值或某些值就表示某種事件，且具有一定的概率；
* 隨機變量中的隨機來源於隨機試驗結果的不確定性。

我們可以通過引入隨機變量，我們簡化了隨機試驗結果（事件）的表示，從而可以更加方便的對隨機試驗進行研究。

**隨機變量的分類**:<br>
* 離散隨機變量；
* 連續隨機變量；
* 每類隨機變量都有其獨特的概率密度函數和概率分佈函數。

**隨機變量的數字特徵**：<br>
* 期望（均值），眾數，分位數，中位數；
* 方差；
* 協方差；
* 相關係數。


### 隨機變量（Random Variable）

*對隨機變量以及其取值規律的研究是概率的核心內容。在上一個小結中，總結了隨機變量的概念以及隨機變量與事件的聯繫。這個小結會更加深入的討論隨機變量。*

#### 隨機變量與事件

隨機變量的本質是一種函數（映射關係），在古典概率模型中，“事件和事件的概率”是核心概念；但是在現代概率論中，“隨機變量及其取值規律”是核心概念。
<br>
**隨機變量與事件的聯繫與區別**
<br>
小結 1 中對著練歌概念的聯繫進行了非常詳細的描述。隨機變量實際上只是事件的另一種表達方式，這種表達方式更加的形式化和符號化，也更佳便於理解以及進行邏輯運算。不同的事件，其實就是隨機變量不同取值的組合。在陈希孺先生的書中，有一個很好的例子來說明這兩者的區別：<br>
> 對於隨機試驗，我們所關心的往往是與所研究的特定問題有關的某個或某些變量，而這些量就是隨機變量。當然，有事我們所關心的是某個或某些特定的隨機時間。例如，在特定一群人中，年收入在萬元以上的高收入者，以及年收入在$3000$元以下的低收入者，各自比率如何？者看上去是兩個孤立的事件。可是當我們引入一個隨機變量$X$： 
> 
> <br>
> <center> $X = $ 隨機抽出一個人其年收入</center>
> <br>
> 
>則$X$是我們關心的隨機變量。上述兩個事件可以分表表示為$\{X > 10000\}$或$\{X < 3000\}$。這就看出：**隨機事件**這個概念實際上包容在**隨機變量**這個更廣的概念之中。也就是說，隨機事件是靜態的觀點來研究隨機現象，而隨機變量則是一種動態的觀點。「概率論能從計算一些孤立事件的概率發展成為一個更高的理論體系，其根本概念就是隨機變量

*這段話，非常清楚的解釋了隨機變量與事件的區別：就跟變量和常量之間的區別那樣*

#### 隨機變量的分類（The Classification of the Random Variable）

**隨機變量從其可能的取值的性質分為兩大類：離散型隨機變量（Discrete Random Variable）和連續性隨機變量（Continuous Random Variable）**<br>
##### 離散型隨機變量（Discrete Random Variable）

離散型隨機變量的取值在整個實數軸上是有間隔的，要麼只有有限個取值，要麼就是無限可數。<br>
如下圖：


```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def poisson_pmf(mu=3):
    
    poisson_dis = stats.poisson(mu)
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    print(x)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu={}) - plot three'.format(mu))
    plt.show()
 
poisson_pmf(mu=8)
```

    [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.  15.
      16.  17.]



![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_19_1.png)



```python
def binom_pmf(n=1, p=0.1):
    binom_dis = stats.binom(n, p)
    x = np.arange(binom_dis.ppf(0.0001), binom_dis.ppf(0.9999))
    print(x)  
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, binom_dis.pmf(x), 'bo', label='binom pmf')
    ax.vlines(x, 0, binom_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of binomial distribution(n={}, p={}) - plot four'.format(n, p))
    
    plt.show()
    

binom_pmf(n=20, p=0.6)
```

    [  4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.  15.  16.  17.  18.]



![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_20_1.png)


* Plot three 是 Poisson Distribution
* Plot four 是 Binomal Distribution

常見的**離散型隨機變量**包括以下這幾種：<br>

* 0-1分佈（Bernoulli Distribution）
* 二項分布（Binomial Distribution）
* 幾何分佈（Geometric Distribution）
* 泊松分佈（Poisson Distribution）
* 超幾何分佈（Hyper-geometric Distribution）



##### 連續型隨機變量（Continuous Random Variable）

連續型隨機變量的取值要麼包括了整個實數集$(-\infty, \infty)$，要麼在一個區間內連續，總之，這一類的隨機變量的可能取值要比離散型隨機變量的取值多得多，個數是無窮不可數的。<br>

**常見的連續型隨機變量包括以下幾種**：<br>

* 均勻分布
* 指數分佈
* 正太分佈（$\gamma$分佈， $\beta$分佈，$\chi^2$分佈等）

##### 概率密度函數的性質

所有的概率密度函數$f(x)$都滿足一下的兩條性質；所有滿足下面兩條性質的一元函數也都可以作為概率密度函數。<br>
$f(x) \geq 0$，以及$\int_{-\infty}^{+\infty}f(x)dx = 1$.

#### 隨機變量的基本性質

隨機變量最主要的性質是其所有可能取到的這些值的取值規律，即取到的概率大小。如果我們把一個隨機變量的所有可能的取值的規律都研究透徹了，那麼這個隨機變量也就研究透徹了。隨機案變量的性質只要有兩類：一類是大而全的性質，這類性質可以詳細描述所有可能取值的概率，律如**累積分佈函數（Cumulative Distribution Funtion）**和**概率密度函數（Probability Density Function）**；另一類是找到該隨機變量的一些特徵或者代表值，例如隨機變量的方差或者期望等數字特徵。常見的隨機變量的性質如下表：


|    name    | 解釋 |
| :---: | :---: |
| CDF: Cumulative Distribution Function |  連續型和離散型隨機變量都有，一般以$F(X)$表示 |
| PDF: Probability Density Function    |  連續型隨機變量在各點的取值規律，用$f(x)$表示 |
| PDF: Probability Density Function    |  連續型隨機變量在各點的取值規律，用$f(x)$表示 |
| PMF: Probability Mass Function    |  離散型型隨機變量在各特定取值上的概率 |
| RVS: Random Variate Sample    |  從一個給定分佈取樣 |
| PPF: Percentile Point Function  |  CDF 的反函數 |
| IQR: Inter Quartile Range   |  $25%$分位數與$75%$ 分位數之差 |

*PDF 只有**連續型隨機變量**才有， PMF 只有**離散型隨機變量**才有；一個分佈的 CDF 求導等於 PDF， 一個分佈的 PDF 積分後就是 CDF*

### 一維離散型隨機變量及其 Python 實現

上一小節，對隨機變量做了一個概述，這一節主要紀錄以為離散變量以及關於他們的一些性質。對於概率論與數理統計方面的計算以及可視化，主要的`pyhton`包有`scipy`，`numpy`和`matplotlib`等。<br>


```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
```

`scipy` 是 `python` 中使用最為廣泛的科學計算工具包，在加上`numpy`和`matplotlib`，基本可以處理大多數的計算和作圖任務。下面是`wikipedia`對`scipy`的介紹：<br>

> SciPy是一个开源的Python算法库和数学工具包。SciPy包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。与其功能相类似的软件还有MATLAB、GNU Octave和Scilab。SciPy目前在BSD许可证下发布。它的开发由Enthought资助。

我們使用的是 `scipy`中的 `stats`模塊，這個模塊包歡樂概率論以及統計相關的函數。
相關函數可以查詢：<a href="https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html"> scipy stats</a>

####  伯努利分佈（Bernoulli Distribution）

又名兩點分佈或者$0-1$分佈，是一個離散型概率分佈。若伯努利試驗成功，則伯努利隨機變量取值為$1$，如果失敗則取值為$0$。記其成功概率為$p(0\leq p \leq 1)$，失敗概率為：$q = 1-p$。其概率質量函數（PMF） 為：<br>
<center>
    $\begin{equation}     \nonumber P_X(x) = \left\{     \begin{array}{l l}     p& \quad \text{for  } x=1\\     1-p & \quad \text{ for } x=0\\     0  & \quad \text{ otherwise }     \end{array} \right. \end{equation}$
    </center>


```python
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)

p = 0.8
x = np.linspace(0,1)

plt.plot(x,bernoulli.pmf(x,p),'o-',label='bernoulli pmf')

plt.title('bernoulli pmf')
plt.xlim(-0.1,1.1)
plt.ylim(0,1)

plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_31_0.png)


伯努利分佈只有一個參數$p$，記做$X ~ Bernuolli(p)$，或$X ~ B(1,p)$，讀做$X$服從參數為$p$的伯努利分佈。伯努利分佈適合於試驗結果只有兩種可能的單次試驗。例如拋一次硬幣，其結果只有正面或者反面兩種可能；一次產品質量檢測，結果只有合格還是不合格這兩種可能。<br>



```python
import scipy.stats as stats
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

def bernoulli_pmf(p=0.0):

    ber_dist = stats.bernoulli(p)
    x = [0, 1]
    x_name = ['0', '1']
    pmf = [ber_dist.pmf(x[0]), ber_dist.pmf(x[1])]
    plt.bar(x, pmf, width=0.15)
    plt.xticks(x, x_name)
    plt.ylabel('Probability')
    plt.title('PMF of bernoulli distribution')
    plt.show()

bernoulli_pmf(p=0.8)
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_33_0.png)


上面兩幅圖都是表示伯努利分佈的 PMF；我們為了得到比較準確的某個服從伯努利分佈的隨機變量的期望，需要大量重複伯努利試驗，例如重複$n$次，然後利用$\frac{正面朝上的次數}{n}$來估計$p$值，當我們重複$n$次以後，這就變成了二項分布，就是下面會提到的**二項分布**。

#### 二项分布（Binomial Distribution）

二項分布是指$n$個獨立的是/非試驗中成功的次數的離散概率分佈，其中每次試驗的成功概率為$p$。這樣的單詞成功/失敗試驗又稱為伯努利試驗。實際上，當$n = 1$時，二項分布就是伯努利分佈。二項分布時顯著性差異的二項試驗的基礎。


```python
import scipy.stats as stats
def binom_pmf(n=1, p=0.1):

    binom_dis = stats.binom(n, p) 
    x = np.arange(binom_dis.ppf(0.0001), binom_dis.ppf(0.9999))
    #print(x)  # [ 0.  1.  2.  3.  4.]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, binom_dis.pmf(x), 'bo',label='binom pmf')
    ax.vlines(x, 0, binom_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of binomial distribution(n={}, p={})'.format(n, p))
    plt.show()

binom_pmf(n=20, p=0.6)
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_36_0.png)


##### 二項分布和其他分佈的關係

1. 二項分布的和<br>
    如果$X~B(n,p)$ 和$Y~B(n,p)$，且$X$和$Y$相互獨立，那麼$X + Y$ 也服從二項分佈：
    $X + Y～B（n+m, p)$
    
2. 伯努利分佈<br>
    二項分布就是$n$重伯努利試驗
    
3. 泊松分佈<br>
   泊松分佈實際上可以通過二項分布推導出來，當$n$很大，$p$很小的時候，我們可以通過極限
   去證明（證明見下方）
   
<br> 我們首先先畫圖來看，當$n = 100，p = 0.1$時


```python
binom_pmf(n=1000000,p=0.00001)
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_38_0.png)



```python
# poisson_distribution_PMF
def poisson_pmf(mu=1):
   
    poisson_dis = stats.poisson(mu)
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    #print(x)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu={})'.format(mu))
    plt.show()

poisson_pmf(mu=10)
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_39_0.png)


由圖可得：兩者近似相等；下面是數學證明：

Let $X$ be as described,
Let $ k \geq 0$ be fixed, we write $p = \frac{\lambda}{n}$ and suppose 
that $n$ is large.<br>
Then: <br>

<center>
    $\begin{align*}
    Pr(X = k) &=  \binom n k p^k \left({1 - p}\right)^{n-k} \\& \simeq \frac {n^k} {k!} \left({\frac \lambda n}\right)^k \left({1 - \frac \lambda n}\right)^n \left({1 - \frac \lambda n}\right)^{-k} \\ &= \frac 1 {k!} \lambda^k \left({1 + \frac {-\lambda} n}\right)^n \left({1 - \frac \lambda n}\right)^{-k} \\ &=  \frac 1 {k!} \lambda^k \left({1 + \frac {-\lambda} n}\right)^n \\ &\simeq \frac{1}{k}\lambda^k e^{-\lambda} \end{align*}$
   </center>
   

when $n \gg k$ it's a reasonable approximation for $\binom n k $, as $ 1-p = (1 - \frac{\lambda}{n})$ is very close to $1$. Hence the result.
<br>
<br>
**Comment:**
Okay wise guy, exactly what constitutes "very large", "very small", and "of a reasonable size"?
Well, if $n = 10^6$ and $p = 10^{-5}$, we have np = 10 = \lambda$
That's the sort of order of magnitude we're talking about here.
    

#### 泊松分佈（Poisson Distribution）

泊松分佈有一個參數$\lambda$（或$\mu$），表示單位事件內隨機事件的平均發生次數，其 PMF 表示為：<br>

<center>
    $\begin{equation}\nonumber P_X(k) = \left\{\begin{array}{l l}\frac{e^{-\lambda} \lambda^k}{k!}& \quad \text{for  } k \in R_X\\       0  & \quad \text{ otherwise} \end{array} \right.             \end{equation}$
</center>
<br>

以上表示單位時間上的泊松分佈，即$t = 1$，如果表示時間$t$上的泊松分佈，則需要將$\lambda$乘以$t$ $\Rightarrow\lambda t$
<br>

一個隨機變量$X$服從參數為$\lambda$的柏松分佈，記做$X～Poisson(\lambda)$，或$X～P(\lambda)$。

泊松分佈適合於描述單位時間內隨機時間發生的次數的概率分佈。如，某一服務設施在一定時間內收到的服務請求的次數，電話交換機接到胡椒的次數，機器出現的故障數，DNA序列的變異數等等。

### 一維連續型隨機變量及其 Python 實現

上一小節總結了幾種離散型隨機變量，這個小節總結連續型隨機變量。離散型隨機變量的可能取值為有限多個或者無限可數，而連續型隨機變量的可能取值則為一段連續的區域或者整個實數軸，是不可數的。最常見的連續型隨機變量有三種：均勻分布、指數分佈和正太分佈。

#### 均勻分佈（Uniform Distribution）

如果連續型隨機變量$X$具有如下的概率目睹函數，則稱$X$服從$[a,b]$上的菊允分佈，記做$X～U[a,b]$<br>
<center>
    $\begin{equation}              \nonumber f_X(x) = \left\{               \begin{array}{l l}                 \frac{1}{b-a} & \quad  a < x < b\\                 0 & \quad x < a \textrm{ or } x > b               \end{array} \right.             \end{equation}$
</center>
<br>

均勻分佈具有等可能性，也就是說服從$U(a,b)$上的均勻分佈的隨機變量$X$落入$(a,b)$中國年的任意子區間的概率只與其取件長度有關，與取件所處的位置無關。<br>

由於是均勻分佈的概率函數是一個常數，因此，其累積分佈函數是一條直線，隨著其取值在定義域內增加，累積分佈函數值均勻增加。
<br>

<center>
    $\begin{equation} 						  \hspace{70pt}                           F_X(x)  = \left\{                           \begin{array}{l l}                             0 & \quad \textrm{for } x < a \\                             \frac{x-a}{b-a} & \quad \textrm{for }a \leq x \leq b\\                             1 & \quad \textrm{for } x > b                           \end{array} \right. 						  \hspace{70pt}                           \end{equation}$
</center>
<br>




```python
from scipy.stats import uniform
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(-2, 2)
ax.plot(x, uniform.cdf(x),'r-', lw=5, alpha=0.6)
plt.title("CDF of uniform distribution")

plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_46_0.png)


均勻分佈主要可以用在：
* 設通過某站的汽車10分鐘一輛，則乘客候車時間$X$，在$[0,10]$上服從均勻分佈；
* 某電台每20分鐘發一個信號，我們隨手打開收音機，等待的時間$X$在$[0,20]$上服從均勻分佈 


```python
def uniform_distribution(loc=0, scale=1):
    """
    均匀分布，在实际的定义中有两个参数，分布定义域区间的起点和终点[a, b]
    :param loc: 该分布的起点, 相当于a
    :param scale: 区间长度, 相当于 b-a
    :return:
    """
    uniform_dis = stats.uniform(loc=loc, scale=scale)
    x = np.linspace(uniform_dis.ppf(0.01),
                    uniform_dis.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.uniform.pdf(x, loc=2, scale=4), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, uniform_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = uniform_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))  # Ture

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Unif({}, {})'.format(loc, loc+scale))
    ax.legend(loc='best', frameon=False)
    plt.show()

uniform_distribution(loc=2, scale=4)
```

    [ 2.004  4.     5.996]
    True



![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_48_1.png)


從定義式中可以看出，定義一個均勻分佈需要兩個參數：定義域的起點$a$和終點$b$，但是在`Python`中是`localtion`是`scale`，分別表示起點和區間長度: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html">scripy.stats.uniform</a><br>

上面的代碼採用了兩種方式$\Rightarrow$直接傳入參數和先凍結了一個分佈，然後畫出均勻分佈的概率分佈函數。此外還從該分佈中選取了10000個值做直方圖。

上圖是一個均勻分佈：$U(2,6)$的概率密度函數曲線

#### 指數分佈（Exponentinal Distribution）

在概率論和統計學中，指數分佈（Exponential Distribution）是一種連續型的概率分佈。可以用來表示獨立隨機時間發生的時間間隔，比如旅客進入機場的時間間隔，打進客服中心電話的時間間隔、中文維基百科新條目出現的時間等等。其實，指數分佈和離散型的泊松分佈有很大關係。泊松分佈表示的是單位時間（或單位面積）內隨機時間的平均發生次數，指數分佈則可以用來表示獨立隨機事件發生的時間間隔。由於發生次數之能事自然數，所以泊松分佈很自然就是離散型的隨機變量；而時間間隔則可以是任意的實數，因此其定義域為：$(0, +\infty)$

如果一個隨機變量$X$的概率密度函數滿足以下形式，就稱$X$為服從參數$\lambda$的指數分佈（Exponential
Distribution），記做$X～E(\lambda)$或$X～Exp(\lambda)$<br>

指數函數只有一個參數$\lambda$，且$\lambda > 0$
<br>
<center>
    $\begin{equation}              \nonumber f_X(x) = \left\{               \begin{array}{l l}                 \lambda e^{-\lambda x} & \quad  x > 0\\                 0 & \quad \textrm{otherwise}               \end{array} \right.             \end{equation}$
</center>
<br>

##### 主要用途：

* 表示獨立隨機案時間發生的時間間隔；
* 在排隊輪中，一個顧客接受服務的時間段也可以用指數分佈來近似；
* 無記憶性的現象（連續時間）

##### 性質

指數分佈的一個顯著的特點就是具有無記憶性。例如如果排隊的顧客接受服務的時間長短服從指數分佈，那麼無論你已經排了多久的隊伍，在排$t$分鐘的概率始終是相同的。
<br>

用公式表達則為：<br>
<center>
    $P(X \geq s + t | X \geq s) = P(X \geq t) \text{  for all  } s,t > 0$
</center>
<br>




```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def exponential_dis(loc=0, scale=1.0):
    """
    指数分布，exponential continuous random variable
    按照定义，指数分布只有一个参数lambda，这里的scale = 1/lambda
    :param loc: 定义域的左端点，相当于将整体分布沿x轴平移loc
    :param scale: lambda的倒数，loc + scale表示该分布的均值，scale^2表示该分布的方差
    :return:
    """
    exp_dis = stats.expon(loc=loc, scale=scale)
    x = np.linspace(exp_dis.ppf(0.000001),
                    exp_dis.ppf(0.999999), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, exp_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = exp_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], exp_dis.cdf(vals)))

    r = exp_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Exp(0.5)')
    ax.legend(loc='best', frameon=False)
    plt.show()

exponential_dis(loc=0, scale=2)
```

    [  2.00100067e-03   1.38629436e+00   1.38155106e+01]
    True



![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_54_1.png)


上圖是，$Exp(0,5)$的概率分佈函數圖<br>

下面是對不同參數的指數分佈的概率分佈函數圖的比較：


```python
def diff_exp_dis():
    """
    不同参数下的指数分布
    :return:
    """
    exp_dis_0_5 = stats.expon(scale=0.5)
    exp_dis_1 = stats.expon(scale=1)
    exp_dis_2 = stats.expon(scale=2)

    x1 = np.linspace(exp_dis_0_5.ppf(0.001), exp_dis_0_5.ppf(0.9999), 100)
    x2 = np.linspace(exp_dis_1.ppf(0.001), exp_dis_1.ppf(0.999), 100)
    x3 = np.linspace(exp_dis_2.ppf(0.001), exp_dis_2.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, exp_dis_0_5.pdf(x1), 'b-', lw=2, label=r'lambda = 2')
    ax.plot(x2, exp_dis_1.pdf(x2), 'g-', lw=2, label='lambda = 1')
    ax.plot(x3, exp_dis_2.pdf(x3), 'r-', lw=2, label='lambda = 0.5')
    plt.ylabel('Probability')
    plt.title(r'PDF of Exponential Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()

diff_exp_dis()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_56_0.png)


#### 正態分佈（Normal Distribution）

正態分佈，又名高斯分佈（Gaussian Distribution），是一種非常常見的連續概率分佈，經常用在自然和社會科學中表示一種不明的隨機變量。由於中心極限定理的存在，正太分佈也是所有分佈中應用最廣泛的分佈。<br>

##### 定義：

若隨機變量$X$的概率密度符合以下形式，就稱$X$服從參數為$\mu, \sigma$的正態分佈，記做：$X～N(\mu,\sigma^2)$.
<br>
<center>
    $f_X (x) = \frac{1}{\sqrt{2 \pi } \sigma} \exp \left\{-\frac{(x - \mu)^2}{2 \sigma^2} \right\}, \hspace{20pt} \textrm{for all } x \in \mathbb{R}.$
</center>
<br>

如果上式公式中$\mu = 0, \sigma = 1$，就叫做標準正態分佈（Standard Normal Distribution），一般記做$Z～N(0,1)$<br>

由於標準正態分佈在統計學中的重要地位，它的累積分佈函數（CDF）有一個專門的表示符號：$\Phi$，一般在統計相關的書籍附錄中的“標準正太分佈函數值表”就是該值與隨機變量的取值之間的對應關係。


```python
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plot

fig, ax = plt.subplots(1, 1)
x = np.linspace(-3,3)
y = stats.norm.cdf(x, loc=0, scale=1)
y1 = stats.norm.pdf(x, loc=0, scale=1)
ax.plot(x,y,'-',label='cdf of standard norm')
ax.plot(x,y1,'-',label='pdf of standard norm')
ax.legend( loc='best',frameon=False)

plt.title(r'Standard Normal Distribution')

plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_58_0.png)


##### 正態分佈兩個參數含義：
* 當固定$\sigma$，改變$\mu$時，$f(x)$圖形的形狀不變，只是沿著$x$軸做平移變換，因此$\mu$被稱為位置參數（決定了（對稱軸的位置）；
*當固定$\mu$，改變$\sigma$時，$f(x)$圖形的對稱軸不變，形狀改變，$\sigma$越小，圖形越高越瘦;$\sigma$越大，圖形越矮越胖，因此$\sigma$被稱為尺度參數（決定曲線的分散程度）<br>

下面是示例：


```python
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plot

fig, ax = plt.subplots(1, 1)
x = np.linspace(-5,5)
y = stats.norm.pdf(x, loc=0, scale=1)
y1 = stats.norm.pdf(x, loc=2, scale=1)
ax.plot(x,y,'-',label='pdf of norm with mu = 0')
ax.plot(x,y1,'-',label='pdf of norm with mu = 2')
ax.legend(loc='best',frameon=True)

plt.title(r'Normal Distribution')

plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_60_0.png)



```python
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plot

fig, ax = plt.subplots(1, 1)
x = np.linspace(-5,5)
y = stats.norm.pdf(x, loc=0, scale=1)
y1 = stats.norm.pdf(x, loc=0, scale=2)
ax.plot(x,y,'-',label='pdf of norm with sigma = 1')
ax.plot(x,y1,'-',label='pdf of norm with sigma = 2')
ax.legend(loc='best',frameon=True)

plt.title(r'Normal Distribution')

plt.show()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_61_0.png)


##### 性質：

* $f(x)$關於$x = \mu$對稱；
* 當$x \leq \mu$時，$f(x)$ 時嚴格單調遞增函數；
* $f_max = f(\mu) = \frac{1}{\sqrt{2\pi}\sigma}$;
* 當$X～N(\mu,\sigma^2)$時，$\frac{X - \mu}{\sigma} \sim N(0, 1)$
<br>

*利用第四點，我們在計算一般的正態分佈時，可以轉化成標準正態分佈進行計算*


```python
def diff_normal_dis():

    norm_dis_0 = stats.norm(0, 1)  # 标准正态分布
    norm_dis_1 = stats.norm(0, 0.5)
    norm_dis_2 = stats.norm(0, 2)
    norm_dis_3 = stats.norm(2, 2)

    x0 = np.linspace(norm_dis_0.ppf(1e-8), norm_dis_0.ppf(0.99999999), 1000)
    x1 = np.linspace(norm_dis_1.ppf(1e-10), norm_dis_1.ppf(0.9999999999), 1000)
    x2 = np.linspace(norm_dis_2.ppf(1e-6), norm_dis_2.ppf(0.999999), 1000)
    x3 = np.linspace(norm_dis_3.ppf(1e-6), norm_dis_3.ppf(0.999999), 1000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x0, norm_dis_0.pdf(x0), 'r-', lw=2, label=r'miu=0, sigma=1')
    ax.plot(x1, norm_dis_1.pdf(x1), 'b-', lw=2, label=r'miu=0, sigma=0.5')
    ax.plot(x2, norm_dis_2.pdf(x2), 'g-', lw=2, label=r'miu=0, sigma=2')
    ax.plot(x3, norm_dis_3.pdf(x3), 'y-', lw=2, label=r'miu=2, sigma=2')
    plt.ylabel('Probability')
    plt.title(r'PDF of Normal Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()

diff_normal_dis()
```


![png](Use_PY_in_Advanced_Statistics%20_files/Use_PY_in_Advanced_Statistics%20_63_0.png)


### 隨機變量的數字特徵（Numerical Characteristic）

如果說一個隨機變量的分佈函數（累積分佈函數或概率密度分佈）是對隨機變量最完整，最具體的描述，那麼隨機變量的數字特徵就是對該隨機變量特徵的描述。分佈函數就如同一個人的全身像，而數字特徵就像一個人的局部特寫。

#### 常見的數字特徵：

* 數學期望（Expectation）
* 方差（Variance）
* 矩（Moments）
* 協方差和相關係數（Covariance and Correlative Coefficient ）

前面三個數字特徵都是耽擱隨機變量自身的特徵，第四個數字特徵則是表示兩個隨機變量之間的關係，其他數學特種還有中位數，眾數等等

#### 數學期望（Matematical Expectation）

一個隨機變量$X$的數學期望，簡稱期望，也叫做均值（Mean），記做E(X)。常見於隨機變量的定義中，都直接或間接包含了“期望”這個參數，該參數一般於分佈在座標軸上的位置有關。期望與我們平時說的平均值不多，體現的是隨機變量中的“大碩鼠”的取值情況或趨勢。
<br>

在計算中，隨機變量$X$的平均值E(X)並不等於一個具體樣本集$x$的均值E(x)$\Rightarrow$計算一個具體樣本集的均值時，是將所有的值求和然後除以樣本個數，因為此時的$x$已經是一個具體的數列，而不再具有隨機性$\Rightarrow$隨機變量$X$的均值是加權平均數。
<br>

例如，一個離散型隨機變量$X$的概率質量分佈列如下：
<br>

|X|0|1|2|3|4|
|------|----|----|----|----|----|
|P(X=x)|0.15|0.30|0.25|0.20|0.10|
<br>

那麼，根據定義，我們可以算出：$E(X) = \displaystyle \sum_{ i = 1 }^{ n } x_i p_i = 0 \times 0.15 + 1 \times 0.3 + 2 \times 0.25 + 3 \times 0.2 + 4 \times 0.1 = 1.8$；如果我們從該隨機變量中取1個樣本集$x_1 = 1，1，2，4，4$，那麼$E(x_1) = \frac{1 + 1 + 2 + 4 + 4}{5} = 2.4$

此外，正是定義中對期望是否存在給出了明確的定義：在求離散型隨機變量的期望時，需要其和式構成的級數是收斂的；在連續型隨機變量時，也有類似的要求。一個典型的例子：連續型隨機變量「柯西分佈」因為不滿足此條件，所以不具有均值，具體解釋可以參考：<a href"http://web.ipac.caltech.edu/staff/fmasci/home/mystats/CauchyVsGaussian.pdf"> Comparing the Cauchy and Gaussian (Normal) density function</a> 和 <a href="https://stats.stackexchange.com/questions/36027/why-does-the-cauchy-distribution-have-no-mean"> Why does the Cauchy distribution have no mean ? </a>

##### 期望的性質：

* 設$c$為一個常數，則$E(c) = c$;
* 設$X$是一個隨機變量，$c$是常數，則$E(cX) = cE(X)$;
* 設$X,Y$是兩個隨機變量，則有$E(X + Y) = E(X) + E(Y)$；<br>

將上面三個性質結合起來，則有：$E(aX + bY + c) = aE(X) + bE(Y) + c$，可以推廣到任意有限個隨機變量線性組合的情況；

* 設$X,Y$是相互獨立的兩個隨機變量，則有$E(XY) = E(X)E(Y)$，可以推廣到任意有限個相互獨立的隨機變量之積的情況。

##### 常見分佈的期望

下面這些分佈的期望是指隨機變量的期望，而不是某個隨機變量抽樣得到的樣本集的期望。在離散隨機變量中，數學期望的物理意義是「一維離散質點系的重心坐標“，在連續型隨機變量中，數學期望的物理意義是「一維連續質點系的重心坐標”。
<br>

* 0-1 分佈：$X～B(1,p), E(X) = p$;
* 二項分佈：$X～B(n,p), E(X) = np$;
* 泊松分佈：$X～P(\lambda), E(X) = \lambda$;
* 幾何分佈：$X～G(p), E(X) = \frac{1}{p}$;
* 均勻分佈：$X～N(a, b), E(X) = \frac{a+b}{2}$;
* 正態分佈：$X～N(\mu, \sigma^2), E(X) = \mu$;
* 指數分佈，$X～E(\lambda), E(X) = \frac{1}{\lambda}$

##### 樣本均值的計算

在實際的應用中，我們一般都是已知某個分佈的一組樣本，需要求這組樣本的均值。在計算時，定義中的平均值時算術平均值；還有一種是計算幾何平均值，及所有樣本值相乘後開$N$次方，$N$為樣本數。
<br> 

算術平均值和幾何平均值最大的區別在於：如果樣本中有$0$存在，幾何平均值就等於$0$；如果樣本中不包含$0$，通常算數平均值$\geq$幾何平均值。下面是 $Python$的實現方法


```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## 计算平均值
x = np.arange(1, 11)
print(x)  # [ 1  2  3  4  5  6  7  8  9 10]
mean = np.mean(x)
print(mean)  # 5.5

# 对空值的处理，nan stands for 'Not-A-Number'
x_with_nan = np.hstack((x, np.nan))
print(x_with_nan)  # [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  nan]
mean2 = np.mean(x_with_nan)
print(mean2)  # nan，直接计算没有结果
mean3 = np.nanmean(x_with_nan)
print(mean3)  # 5.5

## 计算几何平均值
x2 = np.arange(1, 11)
print(x2)  # [ 1  2  3  4  5  6  7  8  9 10]
geometric_mean = stats.gmean(x2)
print(geometric_mean)  # 4.52872868812，几何平均值小于等于算数平均值
```

    [ 1  2  3  4  5  6  7  8  9 10]
    5.5
    [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  nan]
    nan
    5.5
    [ 1  2  3  4  5  6  7  8  9 10]
    4.52872868812


#### 方差（Variance）

一個隨機變量$X$的方差，刻畫了$X$取值的波動性，是衡量該隨機變量取值分散程度的數字特徵。方差越大，就表示該隨機變量越分散；方差越小，表示該隨機變量越集中。在實際應用中，例如常見的「QC」問題，如果一個工廠的出場的合格評分方差大，說明了優質品和劣質品都比較多，**出品不穩定**；相反，如果方差比較小，說明了合格品比較多，優質品和劣質品較少，**出品穩定**

##### 性質：
1. 設$c$為常數，則$Var(c) = 0$；
2. 設$X$是一個隨機變量，$c$是常數，則$Var(cX) = c^2 Var(X)$；特例，$D(-X) = D(X)$；
3. 設$X, Y$是兩個隨機變量，則有$Var(X + Y) = Var(X) + Var(Y) + 2\cdot tail$，其中，$tail = E[X-E(X)][Y-E(Y)]$. 特別的，如果$X, Y$相互獨立，則$tail = 0$
4. $Var(X) = 0 \Leftrightarrow P(X = c) =1,$ 且$c = E(X)$ ;
5. 當$X, Y$相互獨立時，$Var(XY) = Var(X)Var(Y) + Var(X)[E(Y)]^2 +Var(Y)[E(X)]^2$
<br>

還有一個常用的計算方差的公式：$D(X) = E(X^2) - [E(X)]^2$

##### 常見分佈的方差：

* 0-1 分佈：$X～B(1,p), Var(X) = p(1-p)$;
* 二項分佈：$X～B(n,p), Var(X) = np(1-p)$;
* 泊松分佈：$X～P(\lambda), Var(X) = \lambda$;
* 幾何分佈：$X～G(p), Var(X) = \frac{1-p}{p^2}$;
* 均勻分佈：$X～N(a, b), Var(X) = \frac{(b-a)^2}{12}$;
* 正態分佈：$X～N(\mu, \sigma^2), Var(X) = \sigma^2, (\sigma > 0)$;
* 指數分佈，$X～E(\lambda), Var(X) = \frac{1}{\lambda^2}$

##### 樣本方差的計算

就如同計算均值一樣，通常我們的計算都是從某個分佈中抽樣得到的一組樣本的方差，樣本方差一般使用$S^2$表示，按照方差的定義：
<br>
<center>
    $Var(X) = E{[X - E(X)]^2} = \frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \bar{X})^2$
</center>
<br>

其中：$\bar{X} = E(X)$。如果直接用上面的公式計算$S^2$，等同於使用樣本的二階中心距。但是樣本的二階中心距並不是隨機變量$X$這個總體分佈的無偏估計，將上式中的$n$換成$n-1$就得到了樣本方差計算公式，這也是總體方差無偏估計。
<br>
<center>
    $S^2 = \frac{1}{n - 1} \displaystyle \sum_{i=1}^{n}(X_i - \bar{X})^2$
</center>
<br>

從直觀來說，由於樣本方差中多了一個約束條件 ——樣本的均值時固定的，$E(X) = \bar{X} \Rightarrow $如果已知$n-1$樣本，那麼根據均值可以直接計算出第$n$哥樣本的值，因此自由度比計算總體方差的時候減少了$1$個。


```python
import numpy as np

# 参考
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html


data = np.arange(7, 14)
print(data)  # [ 7  8  9 10 11 12 13]

## 计算方差
# 直接使用样本二阶中心距计算方差，分母为n
var_n = np.var(data)  # 默认，ddof=0
print(var_n) # 4.0
# 使用总体方差的无偏估计计算方差，分母为n-1
var_n_1 = np.var(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(var_n_1) # 4.67


## 计算标准差
std_n = np.std(data, ddof=0)
std_n_minus_1 = np.std(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(std_n, std_n_minus_1)  # 2.0, 2.16
print(std_n**2, std_n_minus_1**2)  # 4.0, 4.67
```

    [ 7  8  9 10 11 12 13]
    4.0
    4.66666666667
    2.0 2.16024689947
    4.0 4.66666666667


#### 矩

矩是一個非常廣泛的概念，期望和方差都是矩的特例。<br>

##### 定義：

* 若$E(X^k), k = 1, 2, \dots $存在，則稱$E(X^k)$為$X$的$k$階原點矩，記做$\alpha_k = E(X^k)$；
* 若$E[X - E(X)]^k, k = 1, 2, \dots$存在，則稱$E[X-E(X)]^k$為$X$的$k$階中心距，記做$\beta_k = E[X-E(X)]^k$ 

根據定義，期望$E(X)$ 是1階原點矩，方差$D(X)$是2階中心距。需要注意的是，就跟上面提到的一樣，樣本的2屆中心距並不是總體方差的無偏估計，樣本方差$S^2$的時機計算公式中分母為$n-1$，而不是樣本2階中心距中的$n$。
<br>

**符號說明**：

* 總體k阶原點矩：$α_k$;
* 總體k阶中心矩：$β_k$;
* 樣本k阶原點矩：$A_k$;
* 樣本k阶中心矩：$B_k$;

#### 協方差（Covariance）

上面介紹了幾種隨機變量的數字特徵都是描述耽擱隨機變量拘捕性質的量，協方擦好和相關係數則是用來度量兩個不同的隨機變量之間的相關程度。<br>

##### 協方差

如上面介紹方差的性質時，第(3)條提到的那樣：
<br>

設$X, Y$是兩個隨機變量，則有$ Var(X + Y) = Var(X) + Var(Y) + 2\cdot tail$，其中，$tail = E[X - E(X)][Y - E(Y)].$ 特別的，若$X, Y$相互獨立，那麼$tial = 0$，那麼$tail$不等於$0$，就代表了這兩個隨機變量不相互獨立。$tail$就是$X$與$Y$的協方差。

##### 定義：

數值$E[X-E(Y)][Y-E(Y)]$為隨機變量$X$與$Y$的協方差，記做$Cov(X,Y)$，即：
<br>
<center>
    $Cov(X, Y) = E{[X - E(X)][Y - E(Y)]}$
</center>
<br>

此時，$Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)$，協方差$Cov(X, Y)$反映了隨機變量的$X$與$Y$的線性相關性： <br>

* 當$Cov(X,Y0 > 0$時，稱$X$與$Y$正相關；
* 當$Cov(X,Y0 < 0$時，稱$X$與$Y$負相關；
* 當$Cov(X,Y0 = 0$時，稱$X$與$Y$無關；

協方差的計算公式可以簡化成：$Cov(X,Y) = E(XY) - E(X)E(Y)$

##### 相關係數

協方差時有量綱的數字特徵，為了消除其量綱的影響，引入了相關係數，在平時的數據分析中協方差很少出現，先港係數出現的頻率非常高。
<br>
<center>
    $\rho_{XY} = \frac{Cov(X, Y)}{\sqrt{D(X)D(Y)}}$\
</center>
<br>

稱為隨機變量$X$與$Y$的相關係數。

##### 相關係數的性質

* $|\rho XY|\leq 1$（相關係數的值位於區間[-1,1]）；
* $|\rho XY| = 1 \Rightarrow$ 存在常數$a,b$使得$P(Y = a+bX)=1$. 特別的，$\rho XY = 1$時，$b > 0$；$\rho = 1$時，$b < 0$. 與協方差，相關係數也是用來表徵兩個隨機變量之間線性關係密切程度的特徵數，有時也稱為「線性相關係數」

#### 樣本均值的期望和方差

設隨機變量$X$的一組樣本為$x$，則樣本的矩陣$\bar{x} = \frac{1}{n} \displaystyle \sum_{i=1}^{n}x_i$。此時的樣本$x$與樣本均值$\bar{x}$都是確定的數值，不具有隨機行。但是，如果我們卻了很多組樣本L：$x^{(1)}, x^{(2)}, ...$那麼這些樣本的均值$\bar{x}^{(1)}, \bar{x}^{(2)}, ...$就可以組成一個新的隨機變量，可以記做$\bar{X}$，每一個樣本均值也可以看作從該隨機變量中抽樣所得。<br>

樣本均值$\bar{X}$這個隨機變量時隨機變量$X$的函數。根據期望和方差的定義，我們可以求出樣本均值的期望和方差。<br>
假設隨機變量$X$的期望和方差分別是：$\mu$和$\sigma^2$

* E($\bar{X}$) = $\mu$，樣本均值的期望與原隨機變量的期望相同；
* Var($\bar{X}$) = $\frac{\sigma^2}{n}$，其中$n$為每次取樣的樣本量，



```python
import numpy as np
from scipy import stats

def mean_and_std_of_sample_mean(ss=[], group_n=100):
    """
    不同大小样本均值的均值以及标准差
    """
    norm_dis = stats.norm(0, 2)  # 定义一个均值为0，标准差为2的正态分布
    for n in ss:
        sample_mean = []  # 收集每次取样的样本均值
        for i in range(group_n):
            sample = norm_dis.rvs(n)  # 取样本量为n的样本
            sample_mean.append(np.mean(sample))  # 计算该组样本的均值
        print(np.std(sample_mean), np.mean(sample_mean))

sample_size = [1, 4, 9, 16, 100]  # 每组试验的样本量
group_num = 10000
mean_and_std_of_sample_mean(ss=sample_size, group_n=group_num)
```

    1.99008474793 -0.0208546895623
    1.00750814281 0.00347718092382
    0.672067720702 0.00687662521186
    0.503784014261 0.00284508641179
    0.20016653808 0.000791586359448


#### 樣本均值的期望和方差

設隨機變量$X$的一組樣本為$x$，則樣本的均值$\bar{x} = \frac{1}{n} \displaystyle \sum_{i=1}^{n}x_i$。此時的樣本$x$與樣本均值$x$都是確定的數值，不具有隨機行。但是，如果我們取了很多組樣本：$x^{(1)}, x^{(2)}, ...$那麼這些樣本的均值$\bar{x}^{(1)}, \bar{x}^{(2)}, ...$就可以組成一個新的隨機變量，可以記做$\bar{X}$，每一個樣本均值也可以看做是從該隨機變量中抽樣所得。<br>

樣本均值$\bar{X}$這個隨機變量$X$的函數。根據期望和方差的定義，我們可以求出樣本均值的期望和方差。假設隨機變量$X$的期望和方差分別為$\mu$和$\sigma^2$
* $\E(\bar{X}) = \mu$，樣本均值的期望與原隨機變量的期望相同；
* $D(\bar{X}) = \frac{\sigma^2}{n}$，其中$n$ 為每次取樣的樣本量，這裡的$n$不表示樣本組數，而是單組中的樣本量；樣本組數在這裡並沒有體現。這是因為**$\bar{X}$的随机性是在获取单组样本时体现出来的（即结果的不确定性），跟组数无关（当每组样本获得之后，数据就不在具有随机性了）。由此可见，每次采样的样本量越多，得到的样本均值的方差也越小，也就表示更加准确，但是取样所用的时间和成本也同时增加了。这就需要在准确型和成本之间做一个权衡。<br>

<br>
下面用程序做一個測試，測試的是樣本均值的標準差隨著樣本量的變化而發生的變化，如果方差縮小$n$倍，那麼理論上標準差會縮小$\sqrt n$倍。


```python
import numpy as np
from scipy import stats

def mean_and_std_of_sample_mean(ss=[], group_n=100):
    """
    不同大小样本均值的均值以及标准差
    """
    norm_dis = stats.norm(0, 2)  # 定义一个均值为0，标准差为2的正态分布
    for n in ss:
        sample_mean = []  # 收集每次取样的样本均值
        for i in range(group_n):
            sample = norm_dis.rvs(n)  # 取样本量为n的样本
            sample_mean.append(np.mean(sample))  # 计算该组样本的均值
        print(np.std(sample_mean), np.mean(sample_mean))

sample_size = [1, 4, 9, 16, 100]  # 每组试验的样本量
group_num = 10000
mean_and_std_of_sample_mean(ss=sample_size, group_n=group_num)
```

    1.99183803618 0.0461229296359
    1.00674682296 0.0177540329668
    0.668143024892 0.00299211910315
    0.498323040688 -0.00523627140067
    0.201260608929 0.00134588313727


具體的證明過程，可以參考<a href="https://onlinecourses.science.psu.edu/stat414/node/167">Mean and Variance of Sample Mean</a>
