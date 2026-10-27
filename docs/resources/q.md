---
title: "Fine-tuning LLMs to 1.58bit: extreme quantization made easy"
authors:
  - mohamed-mekkouri
  - marc-sun
  - leandro-von-werra
  - pedro-cuenca
  - omar-sanseviero
  - thomas-wolf
translators:
  - berkan-dogan
  - enes-taha-kocyigit
date_translated: 2026-06-01
original_url: https://huggingface.co/blog/1_58_llm_extreme_quantization
---

# LLM'leri 1.58 Bite İnce Ayar Yapmak: Kuantizasyon Artık Aşırı Kolay

Büyük Dil Modelleri (LLM'ler) boyut ve karmaşıklık bakımından büyüdükçe, hesaplama ve enerji maliyetlerini azaltmanın yollarını bulmak kritik bir sorun hâline gelmiştir. Popüler çözümlerden biri kuantizasyondur; bu yöntemde parametrelerin hassasiyeti standart 16-bit kayan noktalı (FP16) ya da 32-bit kayan noktalı (FP32) formattan 8-bit veya 4-bit gibi daha düşük bit formatlarına indirilir. Bu yaklaşım bellek kullanımını önemli ölçüde azaltıp hesaplamayı hızlandırsa da çoğunlukla doğruluk kaybına yol açar. Hassasiyetin aşırı düşürülmesi, modellerin kritik bilgileri yitirmesine ve dolayısıyla performansın düşmesine neden olabilir.

[BitNet](https://arxiv.org/abs/2402.17764), her parametreyi yalnızca üç değerle temsil eden özel bir transformatör mimarisidur: `(-1, 0, 1)`. Bu sayede parametre başına yalnızca 1,58 yani $log_2(3)$ bitlik aşırı bir kuantizasyon sağlanır. Ancak bunun için modelin sıfırdan eğitilmesi gerekmektedir. Sonuçlar etkileyici olsa da herkesin bir LLM'i önceden eğitecek bütçesi yoktur. Bu kısıtlamanın üstesinden gelmek için, mevcut bir modeli 1,58 bite ince ayar yaparak dönüştürmeye olanak tanıyan birkaç tekniği araştırdık! Nasıl yaptığımızı öğrenmek için okumaya devam edin!

## İçindekiler
- [Özet](#özet)
- [BitNet Daha Ayrıntılı Nedir?](#bitnet-daha-ayrıntılı-nedir)
- [1.58b'de Ön Eğitim Sonuçları](#158bde-ön-eğitim-sonuçları)
- [1.58 Bitte İnce Ayar](#158-bitte-ince-ayar)
- [Özel Çekirdekler ve Kıyaslamalar](#özel-çekirdekler-ve-kıyaslamalar)
- [Sonuç](#sonuç)
- [Teşekkürler](#teşekkürler)
- [Ek Kaynaklar](#ek-kaynaklar)

## Özet

[BitNet](https://arxiv.org/abs/2402.17764), Microsoft Research tarafından tanıtılan ve her parametreyi yalnızca üç değerle temsil eden aşırı kuantizasyon kullanan bir mimaridir: -1, 0 ve 1. Bu sayede parametre başına yalnızca 1,58 bit kullanan bir model elde edilir; bu da hesaplama ve bellek gereksinimlerini önemli ölçüde azaltır.

Bu mimari, matris çarpımı sırasında INT8 toplama hesaplamaları kullanır; oysa LLaMA LLM, FP16 toplama ve çarpma işlemleri kullanır.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/matmulfree.png" alt="The new computation paradigm of BitNet b1.58" style="width: 100%;"/>
  <figcaption>BitNet b1.58'in yeni hesaplama paradigması (kaynak: BitNet makalesi https://arxiv.org/abs/2402.17764)</figcaption>
</figure>

Bu durum, teorik olarak enerji tüketiminin azalmasıyla sonuçlanır: BitNet b1.58, Llama referansına kıyasla matris çarpımı için aritmetik işlem enerjisini 71,4 kat tasarruf etmektedir.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/energy_consumption.png" alt="Energy consumption of BitNet b1.58 compared to LLaMA" style="width: 100%;"/>
  <figcaption>BitNet b1.58'in LLaMA'ya kıyasla enerji tüketimi (kaynak: BitNet makalesi https://arxiv.org/abs/2402.17764)</figcaption>
</figure>

BitNet mimarisini kullanarak bir [Llama3 8B modelini](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) başarıyla ince ayar yaptık ve alt görevlerde güçlü bir performans elde ettik. Geliştirdiğimiz 8B modeller [HF1BitLLM](https://huggingface.co/HF1BitLLM) organizasyonu altında yayımlandı. Bu modellerden ikisi farklı eğitim kurulumlarıyla 10 milyar token üzerinde, üçüncüsü ise 100 milyar token üzerinde ince ayar yapıldı. Özellikle modellerimiz, MMLU kıyaslamalarında Llama 1 7B modelini geride bırakmaktadır.

### Transformers ile Nasıl Kullanılır

BitNet mimarisini Transformers'a entegre etmek için "bitnet" adında yeni bir kuantizasyon yöntemi tanıttık ([PR](https://github.com/huggingface/transformers/pull/33410)). Bu yöntem, standart Linear katmanların, aktivasyonların dinamik kuantizasyonunu, ağırlık açımını ve matris çarpımını destekleyen özel BitLinear katmanlarıyla değiştirilmesini içerir.

Modeli Transformers'ta yüklemek ve test etmek son derece kolaydır; API'de herhangi bir değişiklik yapılmasına gerek yoktur:

```python
model = AutoModelForCausalLM.from_pretrained(
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

input_text = "Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:"

input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
output = model.generate(input_ids, max_new_tokens=10)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

Bu kodla her şey arka planda sorunsuz biçimde yönetilir; ek karmaşıklıklarla uğraşmanıza gerek yoktur; yalnızca transformers'ın en güncel sürümünü yüklemeniz yeterlidir.

Modeli hızlıca test etmek için bu [not defterine](https://colab.research.google.com/drive/1ovmQUOtnYIdvcBkwEE4MzVL1HKfFHdNT?usp=sharing) göz atabilirsiniz.

## BitNet Daha Ayrıntılı Nedir?

[BitNet](https://arxiv.org/abs/2402.17764), Çok Başlıklı Dikkat ve İleri Beslemeli Ağlardaki geleneksel Linear katmanların yerini üçlü hassasiyet (hatta ilk sürümde ikili hassasiyet) kullanan BitLinear adlı özel katmanlarla değiştirir. Bu projede kullandığımız BitLinear katmanları, ağırlıkları üçlü hassasiyetle (-1, 0 ve 1 değerleriyle) kuantize eder; aktivasyonları ise 8-bit hassasiyete kuantize ederiz. Sonraki bölümde göreceğimiz üzere, eğitim ve çıkarım için farklı BitLinear uygulamaları kullanıyoruz.

Üçlü hassasiyetle eğitimin önündeki temel engel, ağırlık değerlerinin ayrıklaştırılmasıdır (`round()` fonksiyonu aracılığıyla); bu nedenle türevlenebilir değildir. BitLinear bunu zarif bir numara olan [STE (Düz Geçiş Tahmincisi)](https://arxiv.org/abs/1903.05662) ile çözer. STE, türev alınamayan yuvarlama işleminden gradyanın geçmesine, gradyanını 1 olarak yaklaşık hesaplayarak (`round()` fonksiyonunu kimlik fonksiyonuna eşdeğer kabul ederek) izin verir. Başka bir bakış açısıyla şöyle ifade edilebilir: gradyanı yuvarlama adımında durdurmak yerine STE, yuvarlama hiç gerçekleşmemiş gibi gradyanın geçmesine izin vererek standart gradyan tabanlı optimizasyon teknikleriyle ağırlık güncellemesi yapılmasını sağlar.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/bitlinear.png" alt="The architecture of BitNet with BitLinear layers" style="width: 100%"/>
  <figcaption>BitLinear katmanlara sahip BitNet mimarisi (kaynak: BitNet makalesi https://arxiv.org/pdf/2310.11453)</figcaption>
</figure>

### Eğitim

Tam hassasiyetle eğitim yaparız; ancak ağırlıkları simetrik, tensor başına kuantizasyon kullanarak üçlü değerlere dönüştürürüz. Önce ağırlık matrisinin mutlak değerlerinin ortalamasını hesaplar ve bunu ölçek olarak kullanırız. Ardından ağırlıkları ölçeğe böler, değerleri yuvarlayarak -1 ile 1 arasına kısıtlar ve tam hassasiyette devam etmek için yeniden ölçeklendiririz.

$$scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|}$$

$$W_q = \text{clamp}_{[-1,1]}(\text{round}(W \times scale_w))$$

$$W_{dequantized} = W_q \times scale_w$$


Aktivasyonlar, belirtilen bit genişliğine (bizim durumumuzda 8-bit) token başına absmax kuantizasyonu kullanılarak kuantize edilir (kuantizasyon yöntemlerine kapsamlı bir giriş için bu [yazıya](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) bakabilirsiniz). Bu işlem, 8-bit genişlik için aktivasyonların `[−128, 127]` aralığına ölçeklenmesini gerektirir. Kuantizasyon formülü şöyledir:


$$scale_x = \frac{127}{|X|_{\text{max}, \, \text{dim}=-1}}$$

$$X_q = \text{clamp}_{[-128,127]}(\text{round}(X \times scale))$$

$$X_{dequantized} = X_q \times scale_x$$


Formülleri daha net hâle getirmek için 3x3 matris kullanan ağırlık ve aktivasyon kuantizasyonu örnekleri aşağıda verilmiştir:

---
<details>
  <summary>Örnek 1: Ağırlık Matrisi Kuantizasyonu</summary>

  Ağırlık matrisi $W$ şöyle olsun:

  $$
  W =
  \begin{bmatrix}
  0.8 & -0.5 & 1.2 \\
  -1.5 & 0.4 & -0.9 \\
  1.3 & -0.7 & 0.2
  \end{bmatrix}
  $$

  **Adım 1: Ağırlıklar İçin Ölçeği Hesapla**
  
  Formülü kullanarak:

  $$scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|}$$

  $W$'nin ortalama mutlak değerini hesaplarız:

  $$\frac{1}{nm} \sum_{ij} |W_{ij}| = \frac{1}{9}(0.8 + 0.5 + 1.2 + 1.5 + 0.4 + 0.9 + 1.3 + 0.7 + 0.2) = \frac{1}{9}(7.5) = 0.8333$$

  Ölçek faktörü şöyle olur:

  $$scale_w = \frac{1}{0.8333} \approx 1.2$$

  **Adım 2: Ağırlık Matrisini Kuantize Et**
  
  Formülü kullanarak:

  $$W_q = \text{clamp}_{[-1, 1]}(\text{round}(W \times scale_w))$$

  Önce ağırlıkları $scale_w \approx 1.2$ ile ölçeklendiririz:

  $$
  W \times scale_w =
  \begin{bmatrix}
  0.8 \times 1.2 & -0.5 \times 1.2 & 1.2 \times 1.2 \\
  -1.5 \times 1.2 & 0.4 \times 1.2 & -0.9 \times 1.2 \\
  1.3 \times 1.2 & -0.7 \times 1.2 & 0.2 \times 1.2
  \end{bmatrix}
  =
  \begin{bmatrix}
  0.96 & -0.6 & 1.44 \\
  -1.8 & 0.48 & -1.08 \\
  1.56 & -0.84 & 0.24
  \end{bmatrix}
  $$

  Ardından değerleri yuvarlayıp $[-1, 1]$ aralığına kısıtlarız:

  $$
  W_q =
  \begin{bmatrix}
  1 & -1 & 1 \\
  -1 & 0 & -1 \\
  1 & -1 & 0
  \end{bmatrix}
  $$

  **Adım 3: Ağırlıkları Tekrar Ölçeklendir**
  
  Son olarak ağırlıkları şu formülle tekrar ölçeklendiririz:

  $$W_{dequantized} = W_q \times scale_w$$

  scale_w'yi yerine koyduğumuzda:

  $$
  W_{dequantized} =
  \begin{bmatrix}
  1 \times 1.2 & -1 \times 1.2 & 1 \times 1.2 \\
  -1 \times 1.2 & 0 \times 1.2 & -1 \times 1.2 \\
  1 \times 1.2 & -1 \times 1.2 & 0 \times 1.2
  \end{bmatrix}
  =
  \begin{bmatrix}
  1.2 & -1.2 & 1.2 \\
  -1.2 & 0 & -1.2 \\
  1.2 & -1.2 & 0
  \end{bmatrix}
  $$

</details>

<details>
  <summary>Örnek 2: Aktivasyon Matrisi Kuantizasyonu</summary>

  Aktivasyon matrisi $X$ şöyle olsun:

  $$
  X =
  \begin{bmatrix}
  1.0 & -0.6 & 0.7 \\
  -0.9 & 0.4 & -1.2 \\
  0.8 & -0.5 & 0.3
  \end{bmatrix}
  $$


  **Adım 1: Aktivasyonlar İçin Ölçeği Hesapla**  

  Her satır (veya kanal) için maksimum mutlak değeri hesaplayın:

  - **Satır 1**: Maksimum mutlak değer = 1.0
  - **Satır 2**: Maksimum mutlak değer = 1.2
  - **Satır 3**: Maksimum mutlak değer = 0.8

  Her satır için ölçek faktörlerini hesaplayın:


  $$
  \text{scale} = \begin{bmatrix}
  \frac{127}{1.0} \\
  \frac{127}{1.2} \\
  \frac{127}{0.8}
  \end{bmatrix}
  =
  \begin{bmatrix}
  127 \\
  105.83 \\
  158.75
  \end{bmatrix}
  $$


  **Adım 2: Aktivasyon Matrisini Kuantize Et**  

  Formülü kullanarak:
  

  $$X_q = \text{clamp}_{[-128,127]}(\text{round}(X \times \text{scale}))$$

  Aktivasyonları ölçeklendirin:

  $$
  X \times \text{scale} =
  \begin{bmatrix}
  1.0 \times 127 & -0.6 \times 127 & 0.7 \times 127 \\
  -0.9 \times 105.83 & 0.4 \times 105.83 & -1.2 \times 105.83 \\
  0.8 \times 158.75 & -0.5 \times 158.75 & 0.3 \times 158.75
  \end{bmatrix}
  =
  \begin{bmatrix}
  127 & -76.2 & 88.9 \\
  -95.2 & 42.3 & -127 \\
  127 & -79.4 & 47.6
  \end{bmatrix}
  $$

  Değerleri yuvarlayıp $[-128, 127]$ aralığına kısıtlayın:

  $$
  X_q =
  \begin{bmatrix}
  127 & -76 & 89 \\
  -95 & 42 & -127 \\
  127 & -79 & 48
  \end{bmatrix}
  $$

  **Adım 3: Aktivasyonları Tekrar Ölçeklendir**  

  Son olarak aktivasyonları şu formülle tekrar ölçeklendirin:

  
  $$X_{dequantized} = X_q \times \frac{1}{\text{scale}}$$

  Ölçekleri yerine koyduğumuzda:

  $$
  X_{dequantized} =
  \begin{bmatrix}
  127 \times \frac{1}{127} & -76 \times \frac{1}{127} & 89 \times \frac{1}{127} \\
  -95 \times \frac{1}{105.83} & 42 \times \frac{1}{105.83} & -127 \times \frac{1}{105.83} \\
  127 \times \frac{1}{158.75} & -79 \times \frac{1}{158.75} & 48 \times \frac{1}{158.75}
  \end{bmatrix}
  =
  \begin{bmatrix}
  1.0 & -0.6 & 0.7 \\
  -0.9 & 0.4 & -1.2 \\
  0.8 & -0.5 & 0.3
  \end{bmatrix}
  $$

</details>

---

Çıktının varyansını korumak amacıyla aktivasyonları kuantize etmeden önce Katman Normalizasyonu (LN) uygularız:


$$\text{LN}(x) = \frac{x - E(x)}{\sqrt{\text{Var}(x) + \epsilon}}$$


Burada ϵ, taşmayı önlemek için kullanılan küçük bir sayıdır.

Daha önce belirtildiği gibi `round()` fonksiyonu türevlenebilir değildir. İleri geçişte türevlenebilir bir düz geçiş tahmincisi uygulamak için `detach()` kullanırız:

```python
# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch
import torch.nn as nn 
import torch.nn.functional as F

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
 
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    """
    Only for training
    """
    def forward(self, x):
        w = self.weight
        x_norm = LN(x)
        
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Perform quantized linear transformation
        y = F.linear(x_quant, w_quant)
        return y
```

### Çıkarım

Çıkarım sırasında ağırlıkları yeniden ölçeklendirmeksizin yalnızca üçlü değerlere kuantize ederiz. Aktivasyonlara 8-bit hassasiyetle aynı yaklaşımı uygular, ardından verimli bir çekirdekle matris çarpımı yapar ve sonucu hem ağırlık hem de aktivasyon ölçeklerine böleriz. Bu işlem, özellikle optimize edilmiş donanımda çıkarım hızını önemli ölçüde artırmalıdır. Matris çarpımlarının düzgün eğitim için fp16/bf16/fp32 formatında tutulması nedeniyle yeniden ölçeklendirme sürecinin eğitim sırasında farklı işlediğini görebilirsiniz.

```python
# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch
import torch.nn as nn 
import torch.nn.functional as F

def activation_quant_inference(x):
    x = LN(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale
 
class BitLinear(nn.Linear):
    """
    Only for training
    """
    def forward(self, x):
        w = self.weight # weights here are already quantized to (-1, 0, 1)    
        w_scale = self.w_scale  
        x_quant, x_scale = activation_quant_inference(x)
        y = efficient_kernel(x_quant, w) / w_scale / x_scale
        return y
```

## 1.58b'de Ön Eğitim Sonuçları

İnce ayar denemelerinden önce BitNet makalesindeki sonuçları ön eğitimle yeniden üretmeye çalıştık. Küçük bir veri seti olan [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories) ve bir [Llama3 8B modeli](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) ile başladık. Makalenin yaptığı gibi bir normalizasyon fonksiyonu eklemenin performansı artırdığını doğruladık. Örneğin 2.000 adım eğitimin ardından, normalizasyon olmadan doğrulama setinde 6,3 olan şaşkınlık (perplexity) değeri normalizasyon ile 5,9'a düştü. Her iki durumda da eğitim kararlı seyretti.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/pre-training.png" alt="Pre-training plots without (blue) & with (green) layer normalisation" style="width: 100%;"/>
  <figcaption>Katman normalizasyonu olmadan (mavi) ve ile (turuncu) ön eğitim grafikleri</figcaption>
</figure>

Bu yaklaşım ön eğitim açısından son derece ilgi çekici görünse de çok az kurum gerekli ölçekte bunu karşılayabilir. Bununla birlikte, halihazırda güçlü önceden eğitilmiş modeller mevcuttur ve bu modellerin ön eğitimin ardından 1,58 bite dönüştürülmesi son derece faydalı olacaktır. Diğer gruplar, ince ayar sonuçlarının ön eğitimle elde edilenler kadar güçlü olmadığını bildirmişti; bu nedenle 1,58 bit ince ayarının işe yarayıp yaramayacağını araştırmaya karar verdik.

## 1.58 Bitte İnce Ayar

Önceden eğitilmiş Llama3 8B ağırlıklarından başlayarak ince ayar yaptığımızda, model biraz daha iyi performans gösterdi ancak beklediğimiz kadar iyi değildi.

> **Not:** Tüm deneylerimiz [Nanotron](https://github.com/huggingface/nanotron) kullanılarak gerçekleştirildi. 1,58 bit ön eğitimi veya ince ayarı denemek isterseniz bu [PR](https://github.com/huggingface/nanotron/pull/180)'a göz atabilirsiniz.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/finetuning_basic.png" alt="Fine-tuning plot compared to pre-training plot" style="width: 100%;"/>
  <figcaption>İnce ayar grafiği ile ön eğitim grafiğinin karşılaştırması</figcaption>
</figure>

Nedenini anlamak için hem rastgele başlatılan modelin hem de önceden eğitilmiş modelin ağırlık dağılımlarını incelemeye çalıştık.

<div style="display: flex; justify-content: center;">
  <figure style="margin-right: 20px; text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/poids_aléatoires.png" alt="Random weights distribution (2 merged stds)" style="width: 400px;" />
    <figcaption>Rastgele ağırlık dağılımı (birleştirilmiş 2 standart sapma)</figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/poids_llama3.png" alt="Pre-trained Llama3 weights distribution" style="width: 400px;" />
    <figcaption>Önceden eğitilmiş Llama3 ağırlık dağılımı</figcaption>
  </figure>
</div>

İki dağılım için ölçek değerleri sırasıyla şöyledir:

<div style="display: flex; justify-content: center;">
  <figure style="margin-right: 20px; text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/scales_random.png" alt="Random weights scales distribution" style="width: 400px;" />
    <figcaption>Rastgele ağırlık ölçekleri dağılımı</figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/scales_llama3.png" alt="Pre-trained Llama3 weights distribution" style="width: 400px;" />
    <figcaption>Önceden eğitilmiş Llama3 ağırlık dağılımı</figcaption>
  </figure>
</div>

Başlangıçtaki rastgele ağırlık dağılımı, iki normal dağılımın karışımıdır:

- Biri $0.025$ standart sapmaya (std) sahip
- Diğeri ise $\frac{0.025}{\sqrt{2 \cdot \text{num\_hidden\_layers}}} = 0.00325$ std değerine sahip

Bu durum, `nanotron`'da sütun doğrusal ve satır doğrusal ağırlıklar için farklı standart sapmalar kullanılmasından kaynaklanır. Kuantize edilmiş versiyonda tüm matrislerin yalnızca 2 ağırlık ölçeği vardır (50,25 ve 402); bunlar her matris için ağırlıkların ortalama mutlak değerinin tersidir: `scale = 1.0 / w.abs().mean().clamp_(min=1e-5)`

- $\text{scale} = 50.25$ için $w.abs().mean() = 0.0199$, bu da $\text{std} = 0.025$ sonucunu verir ve ilk standart sapmamızla örtüşür. Standart sapmayı türetmek için kullanılan formül, $|w|$'nin yarı-normal dağılımının beklenti değerine dayanır:

  $$\mathbb{E}(|w|) = \text{std}(w) \cdot \sqrt{\frac{2}{\pi}}$$

- $\text{scale} = 402$ için $w.abs().mean() = 0.0025$, bu da $\text{std} = 0.00325$ sonucunu verir.

Öte yandan önceden eğitilmiş ağırlık dağılımı, $\text{std} = 0.013$ değerine sahip bir normal dağılıma benzemektedir.

Sonucumuz şuydu: Rastgele ağırlıklardan başlamak modele başlangıçta çok az bilgi verir ve kademeli bir öğrenme sürecine olanak tanır; ince ayar sırasında ise BitLinear katmanlarının devreye girmesi modeli tüm önceki bilgilerini yitirmeye zorlar.

İnce ayar sonuçlarını iyileştirmek için farklı teknikler denedik. Örneğin tensor başına kuantizasyon yerine satır ve sütun başına kuantizasyonu denedik; böylece Llama 3 ağırlıklarından daha fazla bilgi korunabiliyordu. Ayrıca ölçeğin hesaplanma biçimini değiştirmeyi de denedik: ağırlıkların ortalama mutlak değerini ölçek olarak almak yerine aykırı değerlerin ortalama mutlak değerini ölçek olarak aldık (aykırı değer, k*ortalama_mutlak_değer'i aşan bir değerdir; k'yı deneysel olarak değiştirdik). Ancak kayda değer bir iyileşme gözlemlemedik.

```python
def scale_outliers(tensor, threshold_factor=1):
    mean_absolute_value = torch.mean(torch.abs(tensor))
    threshold = threshold_factor * mean_absolute_value
    outliers = tensor[torch.abs(tensor) > threshold]
    mean_outlier_value = torch.mean(torch.abs(outliers))
    return mean_outlier_value

def weight_quant_scaling(w):
    scale = 1.0 / scale_outliers(w).clamp_(min=1e-5)
    quantized_weights = (w * scale).round().clamp_(-1, 1) / scale
    return quantized_weights
```

Hem rastgele ağırlıkların hem de Llama 3 ağırlıklarının kayıplarının yaklaşık 13 gibi aynı değerden başladığını gözlemledik. Bu durum, Llama 3 modelinin kuantizasyon devreye girdiğinde tüm önceki bilgilerini yitirdiğine işaret eder. Bu süreçte modelin ne kadar bilgi yitirdiğini daha ayrıntılı araştırmak için grup başına kuantizasyon deneyleri yaptık.

Akıl yürütme kontrolü olarak önce grup boyutunu 1'e ayarladık; bu da özünde kuantizasyon olmadığı anlamına gelir. Bu senaryoda kayıp 1,45'ten başladı; bu da normal ince ayar sırasında gördüğümüzle aynıdır. Ancak grup boyutunu 2'ye çıkardığımızda kayıp yaklaşık 11'e fırladı. Bu durum, 2 gibi minimal bir grup boyutunda bile modelin hâlâ neredeyse tüm bilgilerini yitirdiğini göstermektedir.

Bu sorunu çözmek için kuantizasyonu her tensöre aniden uygulamak yerine kademeli olarak devreye almanın mümkün olup olmadığını değerlendirdik. Bunun için lambda değeriyle süreci kontrol eden bir uygulama geliştirdik:

```python
lambda_ = ?
x_quant = x + lambda_ * (activation_quant(x) - x).detach()
w_quant = w + lambda_ * (weight_quant(w) - w).detach()
```

`lambda` 0'a ayarlandığında özünde hiçbir kuantizasyon yapılmaz; `lambda=1`'de ise tam kuantizasyon uygulanır.

Başlangıçta 0,25, 0,5, 0,75 ve 1 gibi ayrık `lambda` değerleri test ettik. Ancak bu yaklaşım sonuçlarda kayda değer bir iyileşmeye yol açmadı; çünkü `lambda=0.25` bile kaybın çok yüksekten başlaması için yeterince büyüktü.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/lambda_0.25.png" alt="Fine-tuning plot with lambda = 0.25->0.5->0.75->1" style="width: 100%;"/>
  <figcaption>lambda = 0.25->0.5->0.75->1 ile ince ayar grafiği</figcaption>
</figure>

Bu nedenle eğitim adımına göre dinamik olarak ayarlanan bir `lambda` değeriyle deney yapmaya karar verdik.

```python
lambda_ = training_step / total_training_steps
```

Bu dinamik `lambda` değerinin kullanımı daha iyi kayıp yakınsama sağladı; ancak `lambda` 1'e eşitlendiğinde çıkarım sırasındaki şaşkınlık (ppl) sonuçları hâlâ tatmin edici değildi. Bunun büyük olasılıkla modelin `lambda=1` ile yeterince uzun süre eğitilmemesinden kaynaklandığını fark ettik. Bunu gidermek için `lambda` değerini düzelttik:

```python
lambda_ = min(2 * training_step / total_training_steps, 1)
```

Bu yapılandırmayla 2.000 adım sonrasında şunları elde ettik:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/lambda_training_step.png" alt="Fine-tuning plot with lambda = min(2*training_step/total_training_steps, 1)" style="width: 100%;"/>
  <figcaption>lambda = min(2*training_step/total_training_steps, 1) ile ince ayar grafiği</figcaption>
</figure>

İnce ayar yöntemimiz genel olarak daha iyi yakınsama göstermektedir. Kayıp eğrisinde yaklaşık 1.000 adım civarında hafif bir artış gözlemleyebilirsiniz; bu, `lambda=1`'e yani tam kuantizasyona yaklaşmaya başladığımız noktaya karşılık gelir. Ancak bu noktanın hemen ardından kayıp yeniden yakınsamaya başlar ve şaşkınlık yaklaşık 4'e düşer.

Bu ilerlemeye karşın, kuantize edilmiş modeli WikiText veri setinde (ince ayar için kullandığımız tinystories yerine) test ettiğimizde çok yüksek bir şaşkınlık değeri gözlemledik. Bu durum, modelin düşük-bit modunda belirli bir veri seti üzerinde ince ayar yapılmasının modelin genel bilgisinin büyük bölümünü yitirmesine yol açtığını göstermektedir. Bu sorun, üçlü ağırlıklarla hedeflediğimiz minimal temsillerin veri setinden veri setine önemli ölçüde farklılık göstermesinden kaynaklanıyor olabilir. Bunu çözmek için eğitim sürecimizi daha büyük [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb) veri setini kapsayacak şekilde ölçeklendirdik. `lambda` değerini şu şekilde koruduk:

```python
lambda_ = min(training_step/1000, 1)
```

Bu `lambda` değerini, modeli ısındırmak için iyi bir başlangıç noktası olarak benimsedik. Ardından modeli FineWeb-edu veri seti üzerinde 1e-4 öğrenme hızıyla 5.000 adım boyunca eğittik. Eğitimde 2 milyon toplu iş boyutu (BS) kullandık; toplamda 10 milyar token işledik.

Doğru öğrenme hızını ve doğru azalışı bulmak zordu; bu faktörün modelin performansı açısından kritik önem taşıdığı anlaşılmaktadır.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/fineweb-edu.png" alt="Fine-tuning plot with warmup quantization on Fineweb-edu" style="width: 100%;"/>
  <figcaption>Fineweb-edu üzerinde ısınma kuantizasyonu ile ince ayar grafiği</figcaption>
</figure>

Fineweb-Edu üzerindeki ince ayar sürecinin ardından WikiText veri setindeki şaşkınlık değeri 12,2'ye ulaştı; yalnızca 10 milyar token kullanıldığı göz önünde bulundurulduğunda bu oldukça etkileyici bir sonuçtur. Sınırlı miktarda veriyle diğer değerlendirme metrikleri de güçlü performans sergilemektedir (sonuçlara bakınız).

Lambda 1'e yaklaşırken yaşanan ani artışı yumuşatmayı da denedik. Bunun için başlangıçta üstel büyüyen ardından 1'e yaklaşırken düzleşen lambda zamanlayıcıları kullanmayı düşündük.

```python
def scheduler(step, total_steps, k):
    normalized_step = step / total_steps
    return 1 - (1 - normalized_step)**k
```

Toplam ısınma adımı sayısı 1 olduğunda farklı k değerleri için grafikler şu şekilde görünmektedir:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp_scheduler.png" alt="Exponential scheduler for different k values" style="width: 100%;"/>
  <figcaption>Farklı k değerleri için üstel zamanlayıcı</figcaption>
</figure>

En iyi performans gösteren 1e-4 öğrenme hızını kullanarak k ∈ [4, 6, 8, 10] değerleriyle 4 deney yaptık.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp_scheduler_results.png" alt="Fine-tuning plots with exponential scheduler" style="width: 100%;"/>
  <figcaption>Üstel zamanlayıcı ile ince ayar grafikleri</figcaption>
</figure>

Yumuşatma iyi çalıştı; doğrusal zamanlayıcıda görülen gibi ani artış yaşanmadı. Ancak şaşkınlık tatmin edici değil; ~15 civarında seyretmekte ve alt görevlerdeki performans daha iyi değil.

Ayrıca başlangıçtaki ani artışı da fark ettik; modelin bundan toparlanması güç oldu. lambda = 0 ile özünde hiçbir kuantizasyon yapılmadığından kayıp ~2 gibi düşük bir değerle başlar. Ancak ilk adımın hemen ardından, doğrusal zamanlayıcıda yaşanana benzer bir ani artış görülür (yukarıdaki mavi grafikte olduğu gibi). Bu nedenle farklı bir zamanlayıcı denedik: yavaş başlayan, keskin biçimde 1'e yükselen ve 1'e yaklaşırken düzleşen sigmoid zamanlayıcı.

```python
def sigmoid_scheduler(step, total_steps, k):
    # Sigmoid-like curve: slow start, fast middle, slow end
    normalized_step = step / total_steps
    return 1 / (1 + np.exp(-k * (normalized_step - 0.5)))
```

Farklı k değerleri için aşağıdaki eğriler elde edilmektedir:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/sig_scheduler.png" alt="Sigmoid scheduler for different k values" style="width: 100%;"/>
  <figcaption>Farklı k değerleri için sigmoid zamanlayıcı</figcaption>
</figure>

Bu sefer k ∈ [15, 20, 25, 40, 100] değerleriyle 5 deney yürüttük:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/sig_scheduler_exps.png" alt="Finetuning plots with sigmoid scheduler" style="width: 100%;"/>
  <figcaption>Sigmoid zamanlayıcı ile ince ayar grafikleri</figcaption>
</figure>

Lambda'daki ani artış yaklaşık 500. adımda kararsızlığa yol açtı ve ilk ıraksama sorununu çözmedi. Bununla birlikte $k = 100$ için alt görevlerde bir miktar iyileşme gözledik (sonuç tablosuna bakınız); ancak şaşkınlık ~13,5 civarında kalmaya devam etti. Üstelik doğrusal zamanlayıcıya kıyasla belirgin bir performans artışı da göstermedi.

Buna ek olarak, rastgele ağırlıklar ve çeşitli öğrenme hızlarıyla sıfırdan model eğitimi üzerine de deneyler yaptık. Bu sayede ince ayar yaklaşımımızın etkinliğini geleneksel ön eğitim yöntemleriyle karşılaştırabildik.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp-randoms.png" alt="Different Pre-training plots with different learning rates" style="width: 100%;"/>
  <figcaption>Farklı öğrenme hızlarıyla farklı ön eğitim grafikleri</figcaption>
</figure>

Rastgele ağırlıklarla eğitilen modellerin hiçbiri ince ayar yapılmış modelimizden daha iyi performans göstermedi. Bu modellerle elde ettiğimiz en iyi şaşkınlık değeri 26 oldu; bu sonuç ince ayar yaklaşımımızın sonuçlarının gerisinde kalmaktadır.

### 100 Milyar Tokena Ölçekleme!

Llama 3 8B'nin performansıyla eşleşip eşleşemeyeceğimizi görmek için deneylerimizi 100 milyar tokena ölçeklendirdik. Doğrusal zamanlayıcıyla yapılan kısa çalışmalardan elde ettiğimiz en iyi kontrol noktasından başlayarak 45.000 adım boyunca ince ayara devam ettiğimiz uzun eğitim çalışmaları yürüttük. Farklı öğrenme hızlarını denedik; model bazı metriklerde Llama 3 modeline yakın performans sergilese de ortalamada hâlâ geride kaldı.

Eğitim sırasında çeşitli kontrol noktalarında değerlendirilen metrik örneklerinden bazıları şunlardır:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_100B.png" alt="Metrics evaluations during the training for different lrs" style="width: 100%;"/>
  <figcaption>Farklı öğrenme hızları için eğitim süresince metrik değerlendirmeleri</figcaption>
</figure>

Ortalama puan ise şu şekildedir:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metric_avg.png" alt="Average evaluation during the training for different lrs" style="width: 100%;"/>
  <figcaption>Farklı öğrenme hızları için eğitim süresince ortalama değerlendirme</figcaption>
</figure>

### Daha Küçük Modeller Üzerindeki Deneyler

SmolLM gibi daha küçük modellerle yaptığımız ilk deneylerde, ısınma kuantizasyonu tekniğinin büyük modellerde sağladığı kadar iyileşme sunmadığını gözlemledik. Bu durum, ısınma kuantizasyonunun etkinliğinin model boyutu ve karmaşıklığıyla daha yakından ilişkili olabileceğine işaret etmektedir.

Örneğin [SmolLM 135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) modeli için kayıp eğrileri aşağıda verilmiştir; ısınma kuantizasyonu ile baştan tam kuantizasyon karşılaştırılmaktadır. İlginç biçimde, eğriler büyük ölçüde örtüşmekte ve elde edilen şaşkınlık değerleri de birbirine yakındır.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/smol_llm_exp.png" alt="Smoll LLm fine-tuning experiment with & without warmup quantization" style="width: 100%;"/>
  <figcaption>SmolLM ince ayar deneyi: ısınma kuantizasyonu ile ve olmadan</figcaption>
</figure>

### Sonuçlar ve Karşılaştırma

BitNet, özellikle düşük bit seviyelerinde referans yöntemlere kıyasla güçlü performans sunmaktadır. Makaleye göre BitNet, 8-bit modellerle rekabet edebilecek puanlar elde ederken çıkarım maliyetleri çok daha düşüktür. 4-bit modeller söz konusu olduğunda, yalnızca ağırlıkları kuantize eden yöntemler hem ağırlık hem de aktivasyonları kuantize edenleri geride bırakır; çünkü aktivasyonların kuantize edilmesi daha zordur. Ancak 1,58-bit ağırlıklar kullanan BitNet, hem yalnızca ağırlık kuantizasyonunu hem de ağırlık-aktivasyon kuantizasyonunu geride bırakmaktadır.

Aşağıdaki tablo, Llama3 8B'nin 10 milyar token ile ince ayarının ardından çeşitli metriklerdeki sonuçları sunmaktadır. Bu sonuçlar, performansa kapsamlı bir genel bakış sunmak amacıyla diğer model mimarileriyle karşılaştırılmaktadır. (Tüm değerlendirmeler [Nanotron](https://github.com/huggingface/nanotron) formatındaki model üzerinde [Lighteval](https://github.com/huggingface/lighteval) kullanılarak gerçekleştirildi.)

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_comparison_updated.png" alt="Metrics comparison with Llama models" style="width: 100%;"/>
  <figcaption>Llama modelleriyle metrik karşılaştırması: Linear, Doğrusal lambda zamanlayıcısını; Sigmoid ise Sigmoid lambda zamanlayıcısını (k = 100) ifade eder.</figcaption>
</figure>

Yalnızca 10 milyar token ile üçlü ağırlıklar kullanılarak ince ayar yapılan model, özellikle çok daha kapsamlı eğitimden geçmiş diğer modellerle kıyaslandığında etkileyici bir performans sergilemektedir. Örneğin 100 milyar token gibi çok daha büyük bir veri setiyle eğitilmiş Bitnet 7B modelini geride bırakmaktadır. Ayrıca tam ikilendirilmiş 1,26 trilyon token üzerinde damıtılmış FBI LLM (Fully Binarized LLM) modelinden de daha iyi performans göstermektedir. Bu sonuç, modelin görece küçük ölçekli ince ayar sürecine karşın sergilediği verimlilik ve etkinliği gözler önüne sermektedir.

100 milyar token deneyleri için elimizdeki en iyi performanslı kontrol noktası şu şekildedir:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_100B_table.png" alt="Metrics comparison with Llama models for the model trained on 100B tokens" style="width: 100%;"/>
  <figcaption>100 milyar token ile eğitilmiş model için Llama modelleriyle metrik karşılaştırması</figcaption>
</figure>

Bu sonuçları yeniden oluşturmak için modelleri nanotron formatına dönüştürmek amacıyla bu [PR](https://github.com/huggingface/nanotron/pull/174)'a göz atabilir, ağırlıkları açabilir ([unpack_weights](https://gist.github.com/MekkCyber/78c1532e8767e8da0588b778faf61866) fonksiyonuna bakın) ve lighteval kullanabilirsiniz.

Modellerin Instruct ince ayarlı bir modelden fine-tune edilmiş olmasına karşın bir Instruct veri setiyle de ince ayar yapılmalarına ihtiyaç duyduklarını belirtmek gerekir. Bu modeller, temel model olarak değerlendirilebilir.

## Özel Çekirdekler ve Kıyaslamalar

BitNet'in düşük hassasiyetli ağırlıklarından yararlanmak için bunları `int8` tensörüne paketliyoruz (bu sayede parametre sayısı 8 milyardan 2,8 milyara düşüyor!). Çıkarım sırasında matris çarpımı yapılmadan önce bu ağırlıkların açılması gerekir. Matris çarpımı sürecinde anında açmayı gerçekleştirmek için CUDA ve Triton'da özel çekirdekler uyguladık. Matris çarpımının kendisi için önbelleğe alınmış döşemeli matris çarpımı tekniğini kullandık. Bu yaklaşımı tam olarak kavramak için önce bazı CUDA programlama temellerini gözden geçirelim.

### Temel GPU Kavramları: İş Parçacıkları, Bloklar ve Paylaşılan Bellek

Önbelleğe alınmış döşemeli matris çarpımını ele almadan önce bazı temel GPU kavramlarını anlamak önemlidir:

- **İş Parçacıkları ve Bloklar**: GPU'lar binlerce iş parçacığını eş zamanlı olarak çalıştırır. Bu iş parçacıkları bloklara gruplanır ve her blok bağımsız olarak çalışır. Izgara (grid), bu bloklardan oluşur ve tüm problem uzayını temsil eder. Örneğin matris çarpımında her iş parçacığı, çıktı matrisinin tek bir elemanını hesaplamakla sorumlu olabilir.
- **Paylaşılan Bellek**: Her bloğun, GPU'nun ana belleği olan genel bellekten çok daha hızlı olan sınırlı miktarda paylaşılan belleğe erişimi vardır. Ancak paylaşılan bellek boyut bakımından sınırlıdır ve bir blok içindeki tüm iş parçacıkları tarafından paylaşılır. Paylaşılan belleğin etkin kullanımı, GPU programlarında performansı artırmanın anahtarıdır.

### Matris Çarpımındaki Zorluklar

GPU üzerindeki basit bir matris çarpımı uygulaması, her iş parçacığının sonuç matrisinin tek bir elemanını doğrudan genel bellekten okuyarak hesaplamasını içerebilir. Ancak bu yaklaşım aşağıdaki nedenlerle verimsiz olabilir:

- **Bellek Bant Genişliği**: Genel belleğe erişim, GPU çekirdeklerinin hesaplama yapabildiği hıza kıyasla görece yavaştır. Her iş parçacığı matris elemanlarını doğrudan genel bellekten okursa bellek erişim süreleri bir darboğaza dönüşebilir.
- **Gereksiz Veri Erişimi**: Matris çarpımında girdi matrislerinin birçok elemanı birden fazla kez kullanılır. Her iş parçacığı gerekli verileri genel bellekten bağımsız olarak çekerse aynı veri GPU'ya birden fazla kez yüklenebilir; bu da verimsizliğe yol açar. Örneğin çıktı matrisinin (i, j) konumundaki elemanı hesaplamakla görevli iş parçacığının A matrisinin i'inci satırını ve B matrisinin j'inci sütununu genel bellekten yüklemesi gerekir. Ancak (i+1, j) konumundaki elemanı hesaplayan başka bir iş parçacığı bu veriyi yeniden kullanamaz ve aynı j'inci sütunu genel bellekten yeniden yüklemek zorunda kalır.

### Döşeme Fikri

Döşeme, bu zorlukların üstesinden gelmek için kullanılan bir tekniktir ve temel olarak FlashAttention'da çekirdeğin verimliliğini artırmak amacıyla uygulanmıştır. Temel fikir, matrisleri GPU'nun paylaşılan belleğine sığabilecek boyutlarda alt matrisler (döşemeler) olarak bölmektir. Tüm çıktı matrisini tek seferde hesaplamak yerine hesaplama, döşeme döşeme işlenen daha küçük parçalara ayrılır.

Matris çarpımı bağlamında bu, A ve B matrislerini bloklara (döşemelere) bölmek, bu döşemeleri paylaşılan belleğe yüklemek ve ardından bu küçük bloklarda çarpma işlemini gerçekleştirmek anlamına gelir. Bu yaklaşım, iş parçacıklarının hızlı paylaşılan bellekte depolanan verileri yeniden kullanmasına olanak tanır ve genel belleğe tekrarlı erişim ihtiyacını azaltır.

Nasıl çalıştığı aşağıda özetlenmiştir:

- **Döşemeleri Paylaşılan Belleğe Yükleme**: Her iş parçacığı bloğu, A matrisinden bir döşemeyi ve B matrisinden karşılık gelen bir döşemeyi genel bellekten paylaşılan belleğe işbirliğiyle yükler. Bu işlem döşeme başına bir kez yapılır; ardından döşeme, bloktaki iş parçacıkları tarafından defalarca yeniden kullanılır.
- **Kısmi Çarpımları Hesaplama**: Döşemeler paylaşılan belleğe yüklendikten sonra her iş parçacığı bir kısmi çarpım hesaplar. Bloktaki tüm iş parçacıkları paylaşılan bellekteki aynı döşemeler üzerinde çalıştığından, ek genel bellek erişimine gerek kalmadan verileri verimli biçimde yeniden kullanabilirler.
- **Sonuçları Biriktirme**: Bir döşeme için kısmi çarpımlar hesaplandıktan sonra iş parçacıkları A ve B'den bir sonraki döşemeleri paylaşılan belleğe yükler ve işlemi tekrarlar. Sonuçlar bir yazmaçta (ya da yerel bellekte) biriktirilir; tüm döşemeler işlendikten sonra çıktı matrisinin son değeri genel belleğe geri yazılır.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/illustration_tiling.png" alt="Tiled Matrix multiplication illustration" style="width: 100%;"/>
  <figcaption>Döşemeli Matris Çarpımı gösterimi (kaynak: https://cnugteren.github.io/tutorial/pages/page4.html)</figcaption>
</figure>

**Pratik Değerlendirmeler**

Önbelleğe alınmış döşemeli matris çarpımı uygulanırken birçok faktör göz önünde bulundurulur:

- **Döşeme Boyutu**: Döşeme boyutu, paylaşılan belleğe sığabilecek veri miktarı ile genel bellek erişim sayısı arasındaki dengeyi gözetecek şekilde seçilmelidir.
- **Bellek Birleştirme**: Genel bellek erişimleri birleştirilmelidir; yani bitişik iş parçacıkları bitişik bellek konumlarına erişmelidir.
- **Doluluk**: Blok başına iş parçacığı sayısı ve ızgaradaki blok sayısı, GPU'da mümkün olduğunca fazla etkin warp (bir warp, 32 iş parçacığından oluşan bir kümedir) bulundurmak ve bellek gecikmesini gizlemek için yüksek doluluk sağlayacak şekilde seçilmelidir.

### Triton Çekirdeği

Kıyasladığımız Triton çekirdeği aşağıda verilmiştir:

```python
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for i in range(4) : 
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K) ):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j 

            # BLOCK_SIZE_K must be a divisor of K / 4 
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K // 4 - j * BLOCK_SIZE_K, other=0)
            mask = 3<<(2*i)
            b = ((b_uint8 & mask) >> (2*i))

            # We accumulate the tiles along the K dimension.
            tensor_full = tl.full((1,), 1, dtype=tl.int8)

            accumulator += tl.dot(a, (b.to(tl.int8) - tensor_full), out_dtype=tl.int32)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0] * 4, "Incompatible dimensions, the weight matrix need to be packed"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

### Kod Açıklaması

1. **Döşeme Konumlarının Belirlenmesi**

Çekirdek önce her iş parçacığı bloğunun çıktı matrisinin hangi döşemesinden (bloğundan) sorumlu olduğunu belirler:

- `pid`, `tl.program_id(axis=0)` kullanılarak elde edilen her iş parçacığı bloğunun benzersiz tanımlayıcısıdır.
- Izgara, iş parçacığı bloklarının gruplarına (`GROUP_SIZE_M`) bölünür. Her grup, çıktı matrisinin bir bölümünü işler.
- `pid_m` ve `pid_n`, döşemenin sırasıyla M ve N boyutlarındaki koordinatlarıdır.
- Uzaklıklar (`offs_am`, `offs_bn`, `offs_k`), bloktaki her iş parçacığının A ve B matrislerinin hangi elemanları üzerinde çalışacağını belirlemek için hesaplanır.

2. **Döşemeleri Yükleme ve Hesaplama**

Çekirdek, K boyutunu `BLOCK_SIZE_K` büyüklüğündeki parçalarda yinelemek için bir döngü kullanır. Her parça için:

- **Döşemeleri Yükleme**: A ve B matrislerinden döşemeler genel bellekten yüklenir.
- **B Matrisini Açma**: Çekirdek, B matrisinin `int8` değerleriyle paketlendiğini varsayar; yani her eleman aslında bir bayta sıkıştırılmış dört daha küçük değeri temsil eder. Açma işlemi döngü içinde gerçekleşir:
    - `b_uint8`, genel bellekten paketlenmiş `int8` olarak yüklenir.
    - Her paketlenmiş değer, hesaplamada kullanılan gerçek ağırlık değerlerini elde etmek için açılır.
- **Nokta Çarpımı**: Çekirdek, A ve B'den yüklenen döşemelerin nokta çarpımını hesaplar ve sonuçları `accumulator`'da biriktirir. `accumulator`, C çıktı matrisinin döşemesi için kısmi sonuçları depolar.

3. **Sonuçların Kaydedilmesi**

K boyutundaki tüm döşemeler işlendikten sonra `accumulator`'da saklanan nihai sonuçlar `float16`'ya dönüştürülür ve genel bellekteki C matrisinin karşılık gelen döşemesine geri yazılır. Yazma işlemi, yalnızca geçerli elemanların yazılmasını sağlamak amacıyla bir maske kullanarak bellek sınırlarına uymaktadır.

Kodun daha ayrıntılı bir açıklaması için bu [PR](https://github.com/linkedin/Liger-Kernel/pull/195/files)'a göz atabilirsiniz.

### Kıyaslama

Çekirdeğimizi, `@torch.compile` kullanılarak BF16 hassasiyetinde matris çarpımı gerçekleştiren ağırlık açma yöntemiyle kıyasladık ve her iki yaklaşımın yaklaşık olarak aynı performansı sergilediğini gördük. Doğru kıyaslama sonuçları elde etmek için matris çarpımı işlemini 2.000 kez tekrarladık ve ilk yükleme ya da derlemeyle ilgili verimsizlikleri ortadan kaldırmak amacıyla son 1.000 iterasyondaki sürelerin ortalamasını aldık. Aşağıda kıyaslama sonuçlarını gösteren bir grafik yer almaktadır. Çeşitli matris boyutlarını da test ettik; x ekseni logaritmik ölçekte çarpım sayısını, y ekseni ise milisaniye cinsinden ortalama süreyi göstermektedir.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/without_bitblas.png" alt="Triton kernel compared to torch.compile" style="width: 100%;"/>
  <figcaption>Triton çekirdeği ile torch.compile karşılaştırması</figcaption>
</figure>

Ayrıca karma hassasiyetli matris işlemleri gerçekleştirmek üzere tasarlanmış bir yazılım kütüphanesi olan BitBlas'ı da denedik. Bu kütüphane, hesaplamaların geleneksel FP32 veya FP16 formatları yerine INT8, INT4 hatta INT2 gibi daha düşük hassasiyetli formatlarda yapılmasına olanak tanıyarak bu işlemlerin optimize edilmesine yardımcı olur.

Kıyaslama sonuçları umut verici; grafikte de görüldüğü üzere BitBlas, düşük hassasiyette hem özel çekirdeğimizi hem de Torch'un `matmul` fonksiyonunu geride bırakmaktadır.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/with_bitblas.png" alt="Bitblas benchmark" style="width: 100%;"/>
  <figcaption>Bitblas kıyaslaması</figcaption>
</figure>

Ancak model yüklenirken BitBlas'ın ağırlık matrisinin şekline göre özelleştirilmiş çekirdekler derlemesi ve bunları yerel bir veritabanında depolaması gerekir; bu da ilk yükleme süresini uzatabilir.

## Sonuç

Sonuç olarak, LLM'ler büyümeye devam ettikçe kuantizasyon yoluyla hesaplama taleplerinin azaltılması zorunlu hâle gelmektedir. Bu yazıda üçlü ağırlıklar kullanan 1,58-bit kuantizasyon yaklaşımı incelendi. Modelleri 1,58 bitte ön eğitmek kaynak açısından yoğun bir süreç olsa da bazı tekniklerle mevcut modellerin bu hassasiyet seviyesine ince ayar yapılabileceğini ve doğruluktan ödün vermeden verimli performans elde edilebileceğini gösterdik. Özel çekirdekler aracılığıyla çıkarım hızı optimize edilerek BitNet, LLM'lerin daha pratik ve ölçeklenebilir hâle gelmesi için yeni olanaklar sunmaktadır.

## Teşekkürler
Leandro von Werra, Thomas Wolf ve Marc Sun'a bu proje süresince sağladıkları paha biçilmez destek ve içgörüler için içten teşekkürlerimizi sunarız. Ayrıca blog yazısını iyileştirmemize katkıda bulunan ve bulgularımızı yapay zeka topluluğuna açık ve etkili biçimde aktarmamıza yardımcı olan Omar Sanseviero ve Pedro Cuenca'ya da teşekkür ederiz.
Bunun yanı sıra BitNet projesi üzerindeki öncü çalışmaları için GeneralAI ekibini takdirle anmak istiyoruz. Araştırmaları çabalarımızın temel taşını oluşturmuştur; makalelerinde yer alan net ve özlü figürler için de özellikle minnettarız.

## Ek Kaynaklar
1. H. Wang ve ark., *BitNet: Scaling 1-bit Transformers for Large Language Models*. [arxiv makalesi](https://arxiv.org/pdf/2310.11453)
2. S. Ma ve ark., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. [arxiv makalesi](https://arxiv.org/pdf/2402.17764)
3. S. Ma ve ark., *The Era of 1-bit LLMs: Training Tips, Code and FAQ*. [bağlantı](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
4. RJ. Honicky, *Are All Large Language Models Really in 1.58 Bits?*. [blog yazısı](https://learning-exhaust.hashnode.dev/are-all-large-language-models-really-in-158-bits)
5. L. Mao, *CUDA Matrix Multiplication Optimization*. [blog yazısı](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
6. *Tutorial: OpenCL SGEMM tuning for Kepler*. [bağlantı](https://cnugteren.github.io/tutorial/pages/page4.html)
7. *CUDAMODE*. [github](https://github.com/cuda-mode), [youtube](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
8. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, *Programming Massively Parallel Processors : A Hands-on Approach*
