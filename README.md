# Web UI da metin oluşturma

 GPT-J 6B, OPT, GALACTICA, GPT-Neo, and Pygmalion. Gibi Büyük Dil Modellerini çalıştırmak için gradio web kullanıcı arabirimi

Amacı, [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) metin üretimi.

[[Try it on Google Colab]](https://colab.research.google.com/github/oobabooga/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb)

|![Image1](https://github.com/oobabooga/screenshots/raw/main/qa.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/cai3.png) |
|:---:|:---:|
|![Image3](https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png) | ![Image4](https://github.com/oobabooga/screenshots/raw/main/galactica.png) |

## Özellikler

* Açılır menüyü kullanarak farklı modeller arasında geçiş yapın.
* OpenAI'nin oyun alanına benzeyen not defteri modu.
* Konuşma ve rol oynama için sohbet modu.
* GPT-4chan için güzel HTML çıktısı oluşturun.
* Şunun için Markdown çıktısı oluştur:[GALACTICA](https://github.com/paperswithcode/galai),LaTeX desteği dahil.
* Destek için [Pygmalion](https://huggingface.co/models?search=pygmalionai/pygmalion) ve JSON veya TavernAI Karakter Kartı formatlarında özel karakterler ([FAQ](https://github.com/oobabooga/text-generation-webui/wiki/Pygmalion-chat-model-FAQ)).
* Gelişmiş sohbet özellikleri (resim gönder, TTS ile sesli yanıtlar al).
* Metin çıktısını gerçek zamanlı olarak yayınlayın.
* Metin dosyalarından parametre ön ayarlarını yükleyin.
* Büyük modelleri 8 bit modunda yükleyin ([buraya bakın](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134), [burası](https://github.com/oobabooga/text-generation-webui/issues/20#issuecomment-1411650652) ve [burası](https://www.reddit.com/r/PygmalionAI/comments/1115gom/running_pygmalion_6b_with_8gb_of_vram/) Windows kullanıyorsanız).
* Büyük modelleri GPU'larınız, CPU'nuz ve diskiniz arasında bölün.
* İşlemci modu.
* [FlexGen offload](https://github.com/oobabooga/text-generation-webui/wiki/FlexGen).
* [DeepSpeed ZeRO-3 offload](https://github.com/oobabooga/text-generation-webui/wiki/DeepSpeed).
* API aracılığıyla yanıtları alın, [with](https://github.com/oobabooga/text-generation-webui/blob/main/api-example-streaming.py) or [without](https://github.com/oobabooga/text-generation-webui/blob/main/api-example.py) streaming.
* [RWKV modelini destekler](https://github.com/oobabooga/text-generation-webui/wiki/RWKV-model).
* Yazılım istemlerini destekler.
* [Suzantıları destekler](https://github.com/oobabooga/text-generation-webui/wiki/Extensions).
* [Google Colab'da çalışır](https://github.com/oobabooga/text-generation-webui/wiki/Running-on-Colab).

## Kurulum seçeneği 1: conda

Bir terminal açın ve bu komutları birer birer kopyalayıp yapıştırın (zaten yoksa önce [conda'yı yükleyin](https://docs.conda.io/en/latest/miniconda.html)):

```
conda oluştur -n textgen
conda textgen'i etkinleştir
conda kurulum torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia
git klonu https://github.com/oobabooga/text-generation-webui
cd metin oluşturma-webui
pip kurulumu -r gereksinimleri.txt
```

Üçüncü satır, bir NVIDIA GPU'nuz olduğunu varsayar.

* Bir AMD GPU'nuz varsa, üçüncü komutu bununla değiştirin:

```
pip3 meşaleyi kurun torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
```
  
* CPU modunda çalışıyorsanız, üçüncü komutu bununla değiştirin:

```
conda kurulum pytorch torchvision torchaudio git -c pytorch
```

## Kurulum seçeneği 2: tek tıkla yükleyiciler

[oobabooga-windows.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga-windows.zip)

[oobabooga-linux.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga-linux.zip)

Sadece yukarıdaki zip dosyasını indirin, ayıklayın ve "yükle" ye çift tıklayın. Web kullanıcı arayüzü ve tüm bağımlılıkları aynı klasöre kurulacaktır.

* Bir modeli indirmek için "download-model" üzerine çift tıklayın.
* Web kullanıcı arayüzünü başlatmak için "start-webui" üzerine çift tıklayın

## Modelleri indirme

Modeller, `models/model-name` altına yerleştirilmelidir. Örneğin, [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main) için "models/gpt-j-6B".

#### Hugging Face

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads), modelleri indirmek için ana yerdir. Bunlar bazı dikkate değer örnekler:


* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)
* [GPT-Neo](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads&search=eleutherai+%2F+gpt-neo)
* [Pythia](https://huggingface.co/models?search=eleutherai/pythia)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [\*-Erebus](https://huggingface.co/models?search=erebus) (NSFW)
* [Pygmalion](https://huggingface.co/models?search=pygmalion) (NSFW)

`download-model.py` komut dosyasını kullanarak HF'den otomatik olarak bir model indirebilirsiniz:

    python download-model.py organizasyonu/modeli

Örneğin:

    piton indir-model.py facebook/opt-1.3b

Bir modeli manuel olarak indirmek istiyorsanız, ihtiyacınız olan tek şeyin json, txt ve pytorch\*.bin (veya model*.safetensors) dosyaları olduğunu unutmayın. Kalan dosyalar gerekli değildir.

#### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) Hugging Face'ten kapatıldı, bu yüzden başka bir yerden indirmeniz gerekiyor. İki seçeneğiniz var:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direkt İndirme: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

32 bit sürüm, yalnızca modeli CPU modunda çalıştırmayı düşünüyorsanız geçerlidir. Aksi takdirde, 16 bit sürümünü kullanmalısınız.

Modeli indirdikten sonra şu adımları izleyin:

1. Dosyaları "models/gpt4chan_model_float16" veya "models/gpt4chan_model" altına yerleştirin.
2. GPT-J 6B'nin config.json dosyasını aynı klasöre yerleştirin: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. GPT-J 6B'nin belirteç dosyalarını indirin (GPT-4chan'ı yüklemeye çalıştığınızda bunlar otomatik olarak algılanacaktır):

```
python download-model.py EleutherAI/gpt-j-6B --text-only
```

## Web kullanıcı arayüzünü başlatma

    conda textgen'i etkinleştir
    piton sunucusu.py

Ardından göz atın

`http://localhost:7860/?__theme=dark`



İsteğe bağlı olarak, aşağıdaki komut satırı işaretlerini kullanabilirsiniz:

| Bayrak | Açıklama |
|------------|-------------|
| "-h", "--yardım" | bu yardım mesajını göster ve çık |
| `--model MODEL` | Varsayılan olarak yüklenecek modelin adı. |
| `--defter` | Web kullanıcı arayüzünü, çıktının girişle aynı metin kutusuna yazıldığı not defteri modunda başlatın. |
| `--sohbet` | Web kullanıcı arayüzünü sohbet modunda başlatın.|
| `--cai-sohbet` | Web kullanıcı arayüzünü, Character.AI'ye benzer bir stille sohbet modunda başlatın. `img_bot.png` veya `img_bot.jpg` dosyası server.py ile aynı klasörde bulunuyorsa, bu resim botun profil resmi olarak kullanılacaktır. Benzer şekilde, profil resminiz olarak `img_me.png` veya `img_me.jpg` kullanılacaktır. |
| --işlemci' | Metin oluşturmak için CPU'yu kullanın.|
| `--8bitlik yükleme` | Modeli 8 bit hassasiyetle yükleyin.|
| --bf16' | Modeli bfloat16 hassasiyetiyle yükleyin. NVIDIA Ampere GPU gerektirir. |
| `--oto-cihazlar` | Modeli otomatik olarak mevcut GPU(lar) ve CPU'ya bölün.|
| `-disk` | Model, GPU'larınız ve CPU'nuz için çok büyükse, kalan katmanları diske gönderin. |
| `--disk-cache-dir DISK_CACHE_DIR` | Disk önbelleğinin kaydedileceği dizin. Varsayılanlar "cache/" şeklindedir. |
| `--gpu-bellek GPU_MEMORY [GPU_MEMORY ...]` | GiB'de GPU başına ayrılacak maksimum GPU belleği. Örnek: Tek bir GPU için "--gpu-memory 10", iki GPU için "--gpu-memory 10 5". |
| `--cpu-bellek CPU_MEMORY` | Yükü boşaltılan ağırlıklar için tahsis edilecek GiB'deki maksimum CPU belleği. Tam sayı olmalıdır. Varsayılanlar 99'dur.|
| `--flexgen` | FlexGen boşaltma kullanımını etkinleştirin. |
| `--yüzde YÜZDE [YÜZDE ...]` | FlexGen: ayırma yüzdeleri. Boşluklarla ayrılmış 6 rakam olmalıdır (varsayılan: 0, 100, 100, 0, 100, 0). |
| `--sıkıştır-ağırlık` | FlexGen: Ağırlığın sıkıştırılıp sıkıştırılmayacağı (varsayılan: Yanlış).|
| `--iğne ağırlığı [PIN_WEIGHT]` | FlexGen: ağırlıkların sabitlenip sabitlenmeyeceği (bunun Yanlış olarak ayarlanması CPU belleğini %20 azaltır). |
| `--deepspeed` | Transformers entegrasyonu aracılığıyla çıkarım için DeepSpeed ​​ZeRO-3 kullanımını etkinleştirin. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: ZeRO-3 NVME boşaltması için kullanılacak dizin. |
| `--local_rank LOCAL_RANK` | DeepSpeed: Dağıtılmış kurulumlar için isteğe bağlı bağımsız değişken. |
| `--rwkv-stratejisi RWKV_STRATEGY` | RWKV: Modeli yüklerken kullanılacak strateji. Örnekler: "işlemci fp32", "cuda fp16", "cuda fp16i8". |
| --rwkv-cuda-on' | RWKV: Daha iyi performans için CUDA çekirdeğini derleyin. |
| `--akış yok` | Metin çıktısını gerçek zamanlı olarak yayınlamayın. Bu, metin oluşturma performansını artırır.|
| `--ayarlar SETTINGS_FILE` | Bu json dosyasından varsayılan arayüz ayarlarını yükleyin. Örnek için "settings-template.json"a bakın. `settings.json` adlı bir dosya oluşturursanız, bu dosya `--settings` bayrağını kullanmaya gerek kalmadan varsayılan olarak yüklenir.|
| `--uzantılar UZANTILAR [UZANTILAR ...]` | Yüklenecek uzantıların listesi. Birden fazla uzantı yüklemek istiyorsanız, adları boşluklarla ayırarak yazın. |
| `--dinle` | Web kullanıcı arabiriminin yerel ağınızdan erişilebilir olmasını sağlayın.|
| `--listen-port LISTEN_PORT` | Sunucunun kullanacağı dinleme bağlantı noktası. |
| `--paylaş` | Herkese açık bir URL oluşturun. Bu, web kullanıcı arayüzünü Google Colab veya benzeri bir yerde çalıştırmak için kullanışlıdır. |
| `--ayrıntılı` | Bilgi istemlerini terminale yazdırın. |

Bellek yetersiz hataları? [Bu kılavuzu kontrol edin](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

## Ön ayarlar

Çıkarım ayarları hazır ayarları, "ön ayarlar/" altında metin dosyaları olarak oluşturulabilir. Bu dosyalar başlangıçta otomatik olarak algılanır.

Varsayılan olarak, NovelAI ve KoboldAI tarafından sağlanan 10 ön ayar dahildir. Bunlar, bir K-Means kümeleme algoritması uygulandıktan ve her kümenin ortalamasına en yakın öğeler seçildikten sonra 43 ön ayar örneğinden seçildi.

## Sistem gereksinimleri

Hem GPU hem de CPU modunda VRAM ve RAM kullanımına ilişkin bazı örnekler için [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) sayfasına bakın.

## Katkı

Çekme istekleri, öneriler ve sorun raporları memnuniyetle karşılanır.

Bir hatayı bildirmeden önce, bir conda ortamı oluşturduğunuzdan ve bağımlılıkları tam olarak yukarıdaki *Kurulum* bölümündeki gibi kurduğunuzdan emin olun.

Bu sorunlar bilinmektedir:

* 8-bit, Windows veya daha eski GPU'larda düzgün çalışmaz.
* DeepSpeed, Windows'ta düzgün çalışmıyor.

Bu ikisi için lütfen yeni bir sorun oluşturmak yerine mevcut bir sorun hakkında yorum yapmayı deneyin.

## Kredi

- NovelAI ve KoboldAI ön ayarları: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Pygmalion ön ayarı, sohbet modunda erken durma kodu, bazı kaydırıcılar için kod, --chat modu renkleri: https://github.com/PygmalionAI/gradio-ui/
- Ayrıntılı ön ayar: Anonim 4chan kullanıcısı.
- Instruct-Joi ön ayarı: https://huggingface.co/Rallio67/joi_12B_instruct_alpha
- Gradio açılır menü yenileme düğmesi: https://github.com/AUTOMATIC1111/stable-diffusion-webui
